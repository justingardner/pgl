-- 0015_schema_identity.sql
--
-- Persistent logical identity of this scientific schema.
-- Connection profiles, hostnames, transports, and schema names are not
-- authoritative provenance identities.
--
-- The migration runner owns the transaction and schema search_path.
-- Ordinary application access is SELECT-only, granted by initialization.
--
-- Restore policy:
--   A restored continuation retains this identity.
--   An independently writable clone must receive a new identity through
--   an explicit administrative procedure before diverging.
--   Copying this row does not automatically distinguish writable clones.

CREATE TABLE schema_identity (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    database_id uuid NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT statement_timestamp()
);

-- No INSERT default generates replacement identities during ordinary use.
-- Initialization creates exactly one identity when this migration is applied.
INSERT INTO schema_identity (singleton, database_id)
VALUES (true, pg_catalog.gen_random_uuid());

REVOKE ALL PRIVILEGES ON TABLE schema_identity FROM PUBLIC;

COMMENT ON TABLE schema_identity IS
    'Logical scientific-schema identity. Restored continuations retain it; '
    'independently writable clones require explicit administrative re-identification.';

COMMENT ON COLUMN schema_identity.database_id IS
    'Authoritative UUID for database-qualified checkpoint provenance; '
    'not an endpoint identity or automatic detector of diverging database clones.';
