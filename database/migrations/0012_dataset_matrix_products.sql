-- 0012_dataset_matrix_products.sql
--
-- Register published immutable matrix bundles and attach named products
-- to dataset checkpoints.
--
-- Scientific arrays remain in separately stored Zarr bundles.
-- Source session IDs in provenance do not independently protect sessions.
--
-- No payload garbage collection or matrix soft deletion is introduced.

CREATE TABLE matrix_revisions (
    matrix_revision_id uuid PRIMARY KEY,

    backend_id bigint NOT NULL REFERENCES storage_backends(backend_id),
    object_key text NOT NULL CHECK (pgl_valid_relative_path(object_key)),

    contract_version text NOT NULL DEFAULT 'pgl.matrix.zarr.v1',

    -- Exact JSON descriptor published as matrix.json, including lightweight
    -- wrapper metadata and construction provenance, not scientific arrays.
    descriptor jsonb NOT NULL,

    created_by bigint NOT NULL REFERENCES users(user_id),
    created_at timestamptz NOT NULL DEFAULT now(),

    UNIQUE (backend_id, object_key),

    CONSTRAINT matrix_revisions_contract_check
        CHECK (contract_version = 'pgl.matrix.zarr.v1'),

    CONSTRAINT matrix_revisions_descriptor_object_check
        CHECK (jsonb_typeof(descriptor) = 'object'),

    CONSTRAINT matrix_revisions_descriptor_contract_check
        CHECK (
            (descriptor ->> 'contractVersion')
            IS NOT DISTINCT FROM contract_version
        ),

    CONSTRAINT matrix_revisions_descriptor_identity_check
        CHECK (
            (descriptor ->> 'matrixRevisionID')
            IS NOT DISTINCT FROM matrix_revision_id::text
        )
);

CREATE INDEX matrix_revisions_backend_index
ON matrix_revisions (backend_id);

CREATE TABLE dataset_products (
    dataset_id bigint NOT NULL REFERENCES dataset(dataset_id),

    -- Product names are dataset-local and case-sensitive.
    product_name text COLLATE "C" NOT NULL,

    matrix_revision_id uuid NOT NULL
        REFERENCES matrix_revisions(matrix_revision_id),

    PRIMARY KEY (dataset_id, product_name),

    CONSTRAINT dataset_products_name_check
        CHECK (product_name !~ '^[[:space:]]*$')
);

CREATE INDEX dataset_products_revision_index
ON dataset_products (matrix_revision_id, dataset_id);

-- Matrix catalog rows describe already-published artifacts.
-- A changed matrix requires a new revision and payload path.
--
-- Replica registration and garbage collection are separate future features,
-- not reasons to mutate an existing artifact record.
CREATE FUNCTION pgl_guard_matrix_revision()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
BEGIN
    RAISE EXCEPTION
        'Published matrix revisions are immutable; replacement requires a new revision';
END;
$$;

CREATE TRIGGER matrix_revisions_guard
BEFORE UPDATE OR DELETE ON matrix_revisions
FOR EACH ROW EXECUTE FUNCTION pgl_guard_matrix_revision();

-- Keep dataset lock ordering consistent with membership/publication.
CREATE TRIGGER dataset_products_lock_session_dependencies
BEFORE INSERT OR UPDATE OR DELETE ON dataset_products
FOR EACH STATEMENT
EXECUTE FUNCTION pgl_lock_dataset_session_dependencies();

-- Product links may be assembled only before dataset publication.
-- Soft deletion retains links unchanged.
CREATE FUNCTION pgl_guard_dataset_products()
RETURNS trigger
LANGUAGE plpgsql
SET search_path FROM CURRENT
AS $$
DECLARE
    owner_id bigint;
    owner_status text;
    owner_deleted_at timestamptz;
BEGIN
    IF TG_OP = 'UPDATE'
       AND NEW.dataset_id IS DISTINCT FROM OLD.dataset_id THEN
        RAISE EXCEPTION
            'Products cannot be moved between dataset checkpoints';
    END IF;

    IF TG_OP = 'DELETE' THEN
        owner_id := OLD.dataset_id;
    ELSE
        owner_id := NEW.dataset_id;
    END IF;

    -- Serialize product-link edits against publication of this owner.
    SELECT status, deleted_at
    INTO owner_status, owner_deleted_at
    FROM dataset
    WHERE dataset_id = owner_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Unknown dataset checkpoint: %', owner_id;
    END IF;

    IF owner_status <> 'in_progress' OR owner_deleted_at IS NOT NULL THEN
        RAISE EXCEPTION
            'Products can only change on an active in_progress dataset';
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER dataset_products_guard
BEFORE INSERT OR UPDATE OR DELETE ON dataset_products
FOR EACH ROW EXECUTE FUNCTION pgl_guard_dataset_products();

COMMENT ON TABLE matrix_revisions IS
    'Published immutable Zarr bundle references; registration does not imply continuous integrity verification.';

COMMENT ON TABLE dataset_products IS
    'Immutable named scientific products of a completed dataset checkpoint. Retained during soft deletion.';