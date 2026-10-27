# De-Recursing Grammars

This is a cookbook of examples to help in removing recursion where possible from grammars (see [Syntax](./syntax.md) for more details).
The examples below will generally already be left-recursive.

## Simple lists

```lark
item_list : item
    | item_list item
```
can become
```lark
item_list : item+
```

## Lists with Delimiters

```lark
sep_list : item
    | item_list SEP item
```
becomes
```lark
sep_list : item (SEP item)*
```

## List with alternatives

```lark
postfix_expression: primary_expression
    | postfix_expression "[" expression "]"
    | postfix_expression "(" ")"
    | postfix_expression "(" argument_expression_list ")"
    | postfix_expression "." IDENTIFIER
    | postfix_expression PTR_OP IDENTIFIER
    | "(" type_name ")" "{" initializer_list "}"
    | "(" type_name ")" "{" initializer_list "," "}"
```
becomes (note the additional rule):
```lark
postfix_expression: primary_expression postfix_suffix*
    | "(" type_name ")" "{" initializer_list "}"
    | "(" type_name ")" "{" initializer_list "," "}"

postfix_suffix: "[" expression "]"
    | "(" ")"
    | "(" argument_expression_list ")"
    | "." IDENTIFIER
    | PTR_OP IDENTIFIER
```