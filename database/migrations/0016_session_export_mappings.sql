-- 0016_session_export_mappings.sql
--
-- Destination-local mappings for database-qualified source checkpoints.
-- Historical action/product provenance is never rewritten by this table.
--
-- The migration runner supplies the transaction and search_path.
-- Application access is SELECT/INSERT only.
--
-- Source UUIDs identify logical schemas, not connection profiles.
-- A copied UUID does not distinguish independently diverging clones.
--
-- These records describe verified export/equivalence decisions made by the
-- application. SQL checks destination eligibility and manifest identity;
-- it cannot independently verify source metadata or filesystem bytes.

CREATE TABLE session_export_mappings (
    source_database_id uuid NOT NULL,
    source_session_id bigint NOT NULL CHECK (source_session_id >= 0),
    destination_session_id bigint NOT NULL REFERENCES session(session_id),

    -- Human-readable label captured when this mapping is registered.
    -- Not authoritative for identity or connection selection.
    source_database_name text NOT NULL
        CHECK (btrim(source_database_name) <> ''),

    -- Source and destination manifests may differ after supported
    -- destination-local operational-reference remapping.
    source_manifest_hash char(64) NOT NULL
        CHECK (source_manifest_hash ~ '^[0-9a-f]{64}$'),
    destination_manifest_hash char(64) NOT NULL
        CHECK (destination_manifest_hash ~ '^[0-9a-f]{64}$'),

    source_contract_version text NOT NULL
        CHECK (btrim(source_contract_version) <> ''),
    destination_contract_version text NOT NULL
        CHECK (btrim(destination_contract_version) <> ''),

    -- Versioned verification evidence will be validated by the export
    -- implementation. An arbitrary JSON object is not proof of equivalence.
    verification jsonb NOT NULL
        CHECK (jsonb_typeof(verification) = 'object'),

    registered_by bigint NOT NULL REFERENCES users(user_id),
    registered_at timestamptz NOT NULL DEFAULT statement_timestamp(),

    PRIMARY KEY (source_database_id, source_session_id),

    CONSTRAINT session_export_source_database_nonzero
        CHECK (source_database_id <> '00000000-0000-0000-0000-000000000000'::uuid)
);

CREATE INDEX session_export_mappings_destination_index
ON session_export_mappings (destination_session_id);

-- An insert must point to an active completed destination with precisely
-- the recorded manifest and contract. Hold its row against concurrent
-- lifecycle changes through this transaction.
--
-- Later soft deletion is allowed: mappings remain historical evidence.
-- Lookup/export code must independently enforce current visibility.
CREATE FUNCTION pgl_guard_session_export_mapping()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
DECLARE
    destination_status text;
    destination_deleted_at timestamptz;
    destination_hash text;
    destination_contract text;
BEGIN
    IF TG_OP <> 'INSERT' THEN
        RAISE EXCEPTION
            'Export mappings are immutable; historical mappings cannot be rewritten or deleted';
    END IF;

    SELECT status, deleted_at, manifest_hash, contract_version
    INTO destination_status, destination_deleted_at,
         destination_hash, destination_contract
    FROM session
    WHERE session_id = NEW.destination_session_id
    FOR SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Export destination checkpoint % does not exist',
            NEW.destination_session_id;
    END IF;

    IF destination_status <> 'complete' OR destination_deleted_at IS NOT NULL THEN
        RAISE EXCEPTION
            'Export destination checkpoint % must be complete and active',
            NEW.destination_session_id;
    END IF;

    IF destination_hash IS DISTINCT FROM NEW.destination_manifest_hash::text
       OR destination_contract IS DISTINCT FROM NEW.destination_contract_version THEN
        RAISE EXCEPTION
            'Export destination checkpoint manifest or contract differs from the mapping';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER session_export_mappings_guard
BEFORE INSERT OR UPDATE OR DELETE ON session_export_mappings
FOR EACH ROW EXECUTE FUNCTION pgl_guard_session_export_mapping();

REVOKE ALL PRIVILEGES ON TABLE session_export_mappings FROM PUBLIC;

COMMENT ON TABLE session_export_mappings IS
    'Immutable destination-local source-qualified checkpoint mappings. '
    'Multiple sources may map to one verified destination. '
    'Mappings remain after destination soft deletion; callers enforce visibility.';

COMMENT ON COLUMN session_export_mappings.verification IS
    'Application-validated versioned verification evidence. '
    'SQL does not establish source authenticity or payload equivalence.';
