"""Tests for _clear_datastore_resource's privilege fallback.

xloader empties an existing datastore table with::

    TRUNCATE TABLE "<resource_id>" RESTART IDENTITY

TRUNCATE needs the TRUNCATE privilege on the table, and RESTART IDENTITY
additionally needs ownership of the table's _id sequence. Where the datastore
write role holds neither, PostgreSQL raises InsufficientPrivilege (SQLSTATE
42501) and the whole load fails.

Seen in amplus-data as three shapes of the same fault:

    permission denied for table 65dd096f-7296-40e8-8cfe-e26b928bcce5
    must be owner of table 530cc972-3ac8-4751-9a41-81bd17f17...
    must be owner of sequence ...

DELETE FROM needs only the DELETE privilege, so the loader now falls back to
it. These tests pin that behaviour without needing a database: the engine is
a stub that records the SQL it was handed.
"""
import psycopg2
import pytest
import sqlalchemy as sa

from ckanext.xloader import loader


RESOURCE_ID = "65dd096f-7296-40e8-8cfe-e26b928bcce5"


def denied():
    """A SQLAlchemy wrapper around SQLSTATE 42501, as psycopg2 raises it."""
    return sa.exc.ProgrammingError(
        "TRUNCATE", {},
        psycopg2.errors.InsufficientPrivilege(
            "permission denied for table {0}".format(RESOURCE_ID)))


def syntax_error():
    """A ProgrammingError that is *not* a privilege problem."""
    return sa.exc.ProgrammingError(
        "TRUNCATE", {}, psycopg2.errors.SyntaxError("syntax error"))


class FakeEngine(object):
    """Minimal stand-in for a SQLAlchemy engine, recording executed SQL.

    ``fail_on`` is a substring; any statement containing it raises whatever
    ``error`` returns.
    """

    def __init__(self, fail_on=None, error=denied):
        self.statements = []
        self.fail_on = fail_on
        self.error = error

    def begin(self):
        engine = self

        class _Conn(object):
            def execute(self, clause):
                sql = str(clause)
                engine.statements.append(sql)
                if engine.fail_on and engine.fail_on in sql:
                    raise engine.error()

        class _Ctx(object):
            def __enter__(self):
                return _Conn()

            def __exit__(self, *exc):
                return False

        return _Ctx()


@pytest.fixture
def install_engine(monkeypatch):
    def _install(**kwargs):
        engine = FakeEngine(**kwargs)
        monkeypatch.setattr(loader, "get_write_engine", lambda: engine)
        return engine

    return _install


def test_truncate_is_used_when_permitted(install_engine):
    """The fast path must stay the fast path."""
    engine = install_engine()

    loader._clear_datastore_resource(RESOURCE_ID)

    assert any("TRUNCATE TABLE" in s for s in engine.statements)
    assert not any("DELETE FROM" in s for s in engine.statements)


def test_delete_is_used_when_truncate_is_denied(install_engine):
    """InsufficientPrivilege on TRUNCATE must fall back, not fail the job."""
    engine = install_engine(fail_on="TRUNCATE TABLE")

    loader._clear_datastore_resource(RESOURCE_ID)

    assert any('DELETE FROM "{0}"'.format(RESOURCE_ID) in s
               for s in engine.statements)


def test_lock_timeout_is_set_on_the_fallback_too(install_engine):
    """The fallback holds the same 15s lock_timeout as the fast path."""
    engine = install_engine(fail_on="TRUNCATE TABLE")

    loader._clear_datastore_resource(RESOURCE_ID)

    assert len([s for s in engine.statements if "lock_timeout" in s]) == 2


def test_other_programming_errors_still_propagate(install_engine):
    """Only SQLSTATE 42501 is caught; anything else must surface."""
    install_engine(fail_on="TRUNCATE TABLE", error=syntax_error)

    with pytest.raises(sa.exc.ProgrammingError):
        loader._clear_datastore_resource(RESOURCE_ID)
