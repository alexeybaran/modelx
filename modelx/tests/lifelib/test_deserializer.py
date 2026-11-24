import pathlib
from io import StringIO
import tokenize

import pytest
from modelx.serialize import deserializer
from modelx.serialize.serializer_6 import SourceStructure
from modelx.serialize.ziputil import read_str_utf8
import asttokens


def get_statmetns_atok(path):
    txt = pathlib.Path(path).read_text(encoding='utf-8')
    atok = asttokens.ASTTokens(txt, parse=True)
    return list(atok.get_text(stmt) for stmt in atok.tree.body)


def without_trailing_inline_comment(statement):
    tokens = [token for token in tokenize.generate_tokens(StringIO(statement).readline)
              if token.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER,
                                    tokenize.INDENT, tokenize.DEDENT)]
    if tokens and tokens[-1].type == tokenize.COMMENT:
        row, column = tokens[-1].start
        lines = statement.splitlines(keepends=True)
        before = lines[row - 1][:column]
        if before.strip():
            return ''.join(lines[:row - 1]) + before.rstrip(' \t')
    return statement


def test_compare_statements_agains_asttokens_using_lifelib():
    import lifelib

    lib_dir = pathlib.Path(lifelib.__file__).parent / 'libraries'
    file_count = 0

    for path_ in lib_dir.rglob('__init__.py'):
        print(path_)
        srcstruct = SourceStructure(read_str_utf8(path_))
        file_count += 1
        modelx_statements = list(deserializer.get_statement_tokens(path_))
        asttokens_statements = get_statmetns_atok(path_)
        assert len(modelx_statements) == len(asttokens_statements), path_
        for t, a in zip(modelx_statements, asttokens_statements):
            assert without_trailing_inline_comment(t.str_) == without_trailing_inline_comment(a)
            assert srcstruct.get_section(t.lineno[0]) == t.section

    assert file_count > 30
