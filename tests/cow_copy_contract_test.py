"""Source-extracted CPU copy contracts; no Torch/native/model import.

Use --site after source composition. Fake byte rows test host routing/snapshot
semantics only; native nondefault-stream byte tests are separate evidence.
"""
import argparse
import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--site',type=Path);parser.add_argument('--baseline',type=Path)
ARGS,REST=parser.parse_known_args()
REL='vllm/v1/worker/utils.py'


@dataclass(frozen=True)
class Device:
    type:str='cuda'


class Indices:
    def __init__(self,pairs,device=Device('cpu')):self.pairs=[list(p)for p in pairs];self.device=device
    def tolist(self):return copy.deepcopy(self.pairs)
    def unbind(self,dim):return [p[0]for p in self.pairs],[p[1]for p in self.pairs]


class Storage:
    def __init__(self,nonraw=False):
        self.rows=[[10*i+j for j in range(4)]for i in range(8)]
        self.nonraw=nonraw
    def data_ptr(self):return id(self)
    def nbytes(self):return 32+int(self.nonraw)


class Blocks:
    def __init__(self,storage=None,device=Device()):self.storage=storage;self.device=device
    def set_(self,storage):self.storage=storage;return self
    def view(self,*shape):return self
    def is_contiguous(self):return True
    def __getitem__(self,key):return [self.storage.rows[i].copy()for i in key]
    def __setitem__(self,key,values):
        for i,value in zip(key,values):self.storage.rows[i]=value.copy()


class Cache:
    shape=(8,4)
    def __init__(self,storage,offset=0,device=Device()):self.storage=storage;self.offset=offset;self.device=device
    def data_ptr(self):return self.storage.data_ptr()+self.offset
    def untyped_storage(self):return self.storage
    def stride(self,dim):return 4
    def element_size(self):return 1
    def unflatten(self,*args):return Blocks(self.storage,self.device)


def source_function(path,native_available=True):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='copy_kv_cache_blocks_inplace')
    counts={'native':0,'H2D':0}
    def native(src,dst,size,mapping):
        counts['native']+=1
        assert size==4 and mapping.device.type=='cpu'
        for a,b in mapping.tolist():dst.storage.rows[b]=src.storage.rows[a].copy()
    namespace=NS()
    if native_available:namespace.swap_blocks=native
    torch=NS(uint8=object(),ops=NS(_C_cache_ops=namespace),
        empty=lambda *a,**kw:Blocks(device=kw['device']),from_numpy=lambda x:x)
    def h2d(value,device):
        counts['H2D']+=1
        return Indices(value.tolist(),device)
    scope={'torch':torch,'np':NS(int64=object(),array=lambda x,**kw:Indices(x)),'async_tensor_h2d':h2d}
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'source-extracted-copy-host-contract','exec'),scope)
    return scope['copy_kv_cache_blocks_inplace'],counts


@unittest.skipUnless(ARGS.site,'Pass --site for actual source-extracted host contracts')
class CopyHostContracts(unittest.TestCase):
    def compare(self,pairs,*,native=True,nonraw=False,aliases=False,device=Device()):
        candidate,counts=source_function(ARGS.site/REL,native)
        actual=Storage(nonraw);expected=Storage(nonraw)
        def views(s):
            first=Cache(s,device=device)
            return [first,Cache(s,1,device),first]if aliases else[first]
        if ARGS.baseline:
            reference,_=source_function(ARGS.baseline/REL,native)
            reference(views(expected),8,pairs)
        else:
            values=[expected.rows[a].copy()for a,b in pairs]
            for (_,b),value in zip(pairs,values):expected.rows[b]=value
        candidate(views(actual),8,pairs)
        self.assertEqual(actual.rows,expected.rows)
        return counts

    def test_native_disjoint_repeated_source_no_H2D_indices(self):
        self.assertEqual(self.compare([(0,4),(0,5),(1,6)]),{'native':1,'H2D':0})

    def test_dependent_chain_preserves_original_RHS_snapshot(self):
        self.assertEqual(self.compare([(0,1),(1,2)]),{'native':0,'H2D':1})

    def test_duplicate_destination_retains_legacy_route(self):
        # Fake rows check routing only; old real duplicate writes can race.
        self.assertEqual(self.compare([(0,2),(1,2)]),{'native':0,'H2D':1})

    def test_missing_native_retains_original_gather(self):
        self.assertEqual(self.compare([(0,4),(1,5)],native=False),{'native':0,'H2D':1})

    def test_distinct_alias_views_copy_shared_storage_once(self):
        self.assertEqual(self.compare([(0,4),(1,5)],aliases=True),{'native':1,'H2D':0})

    def test_nonraw_view_retains_original_gather(self):
        self.assertEqual(self.compare([(0,4),(1,5)],nonraw=True),{'native':0,'H2D':1})

    def test_negative_source_and_CPU_storage_keep_legacy_route(self):
        self.assertEqual(self.compare([(-1,0)]),{'native':0,'H2D':1})
        self.assertEqual(self.compare([(0,4)],device=Device('cpu')),{'native':0,'H2D':1})

    def test_empty_plan_noop(self):
        self.assertEqual(self.compare([]),{'native':0,'H2D':0})


if __name__=='__main__':unittest.main(argv=[__file__,*REST])
