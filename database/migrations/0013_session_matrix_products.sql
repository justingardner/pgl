-- 0013_session_matrix_products.sql
--
-- Attach named immutable matrix revisions to session checkpoints.
-- Scientific arrays remain in separately stored Zarr bundles.
--
-- Links may change only while their owning session is active and in_progress.
-- Completed checkpoints retain their links, including during soft deletion.
--
-- No payload garbage collection or matrix soft deletion is introduced.

CREATE TABLE session_products (
    session_id bigint NOT NULL REFERENCES session(session_id),

    -- Product names are session-local and case-sensitive.
    product_name text COLLATE "C" NOT NULL,

    matrix_revision_id uuid NOT NULL
        REFERENCES matrix_revisions(matrix_revision_id),

    PRIMARY KEY (session_id, product_name),

    CONSTRAINT session_products_name_check
        CHECK (product_name !~ '^[[:space:]]*$')
);

CREATE INDEX session_products_revision_index
ON session_products (matrix_revision_id, session_id);

CREATE FUNCTION pgl_guard_session_products()
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
       AND NEW.session_id IS DISTINCT FROM OLD.session_id THEN
        RAISE EXCEPTION
            'Products cannot be moved between session checkpoints';
    END IF;

    IF TG_OP = 'DELETE' THEN
        owner_id := OLD.session_id;
    ELSE
        owner_id := NEW.session_id;
    END IF;

    -- Serialize attachment changes against completion or deletion of the
    -- owning checkpoint. The completion path also locks this session row.
    SELECT status, deleted_at
    INTO owner_status, owner_deleted_at
    FROM session
    WHERE session_id = owner_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Unknown session checkpoint: %', owner_id;
    END IF;

    IF owner_status <> 'in_progress' OR owner_deleted_at IS NOT NULL THEN
        RAISE EXCEPTION
            'Products can only change on an active in_progress session';
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER session_products_guard
BEFORE INSERT OR UPDATE OR DELETE ON session_products
FOR EACH ROW EXECUTE FUNCTION pgl_guard_session_products();

COMMENT ON TABLE session_products IS
    'Immutable named scientific products of a completed session checkpoint. Retained during soft deletion.';

COMMENT ON COLUMN session_products.matrix_revision_id IS
    'Exact published matrix revision; does not imply that an exposed working matrix remains unchanged.';