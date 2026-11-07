from typing import List, Union, Dict, Any
from dataclasses import dataclass
import json
import sys


class StructTag:
    """
    Represents a structural tag used for constraining function/tool invocations in the middle
    of running text.

    These tags are just syntactic sugar over lazy lexemes or special tokens.
    See also https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#tool-calling

    Attributes:
        trigger (str): A substring or token that signals when to start applying this tag.
            For example "<function" or "<|python_tag|>"
        begin (str): The beginning of the tag. Must start with trigger.
            For example "<function=foo>" or '<|python_tag|>{"name":"foo","parameters":'
        grammar (Union[str, dict]): The grammar definition for the arguments of function.
            It can be JSON schema (stringified or as a dict),
            or a full Lark grammar (with 'start:' rule).
        end (str): The string to force at the end of the structured segment.
            For example: "</function>", "}", ""
    """

    def __init__(self, *, trigger: str, begin: str,
                 grammar: Union[str, Dict[str, Any]], end: str):
        self.trigger = trigger
        self.begin = begin
        self.grammar = grammar
        self.end = end
        self._validate()

    def _validate(self) -> None:
        assert len(self.trigger) > 0, "trigger must not be empty"
        assert self.begin.startswith(
            self.trigger), "begin must start with trigger"

    @staticmethod
    def to_grammar(
        tags: List['StructTag'],
        *,
        assume_special: bool = True,
        text_regex: str = r"(.|\n)*",
    ) -> str:
        """
        Generates a Lark grammar string based on the provided structural tags.

        Arguments:
        tags: List[StructTag]: A list of structural tags to generate grammar for.
        assume_special: bool: A flag indicating whether to assume triggers of the format <...> are in fact special tokens.
            Defaults to true.
        text_regex: str: A regex pattern for matching text segments, defaults to r"(.|\n)*", which allows all strings.
        """

        def gtext(s: str) -> str:
            if s:
                return json.dumps(s)
            else:
                return ""

        assert len(tags) > 0, "tags must not be empty"
        assert "/" not in text_regex, "text_regex must not contain /"
        for tag in tags:
            assert isinstance(tag,
                              StructTag), "tags must be StructTag instances"
            tag._validate()
        tag_options = " | ".join(f"tag_{i}" for i in range(len(tags)))
        lark = f"""
%llguidance {{}}
start: ({tag_options})* tag_end
tag_end: TAG_TEXT
TAG_TEXT: /{text_regex}/
"""
        side_grammars = []

        for tag_idx, tag in enumerate(tags):
            lark += "\n"
            tag_rule = f"tag_{tag_idx}"
            trig = tag.trigger

            if isinstance(tag.grammar, str):
                if tag.grammar.lstrip().startswith("{"):
                    grm = "%json " + tag.grammar
                else:
                    gname = f"{tag_rule}_grm"
                    side_grammars.append({
                        "name": gname,
                        "lark_grammar": tag.grammar
                    })
                    grm = "@" + gname
            elif isinstance(tag.grammar, dict):
                grm = "%json " + json.dumps(tag.grammar)
            else:
                raise ValueError("grammar must be a string or a dictionary")

            beg = tag.begin[len(trig):]
            body = f"{gtext(beg)} {grm} {gtext(tag.end)}"

            if assume_special and trig.startswith("<") and trig.endswith(">"):
                # f_qux: TEXT <|placeholder1|> "qux(" /[0-9]+/ ")"
                lark += f"{tag_rule}: TAG_TEXT {trig} {body}\n"
            else:
                # f_baz_hd[lazy]: TEXT "<tool"
                # f_baz: f_baz_hd "=baz>" /[0-9]+/ "</tool>"
                lark += f"{tag_rule}_trig[lazy]: TAG_TEXT {gtext(trig)}\n"
                lark += f"{tag_rule}: {tag_rule}_trig {body}\n"

        lark = lark.lstrip()
        if side_grammars:
            side_grammars.insert(0, {
                "name": "struct_tag",
                "lark_grammar": lark
            })
            return json.dumps({"grammars": side_grammars})
        else:
            return lark
