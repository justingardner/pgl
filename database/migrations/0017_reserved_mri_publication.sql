-- Durable raw-MRI publication. No GC or resume implementation.
DO $pgl_install_publication$
DECLARE
    target_schema text := current_schema();
    definition text := $pgl_publication_definition$
-- Draft only. Not installed by the migration runner.
--
-- IDs are allocated from the existing session identity sequence.
-- Reservations deliberately have no FK to session: they precede it
-- and remain as accounting records after checkpoint purge.
--
-- A later publisher must INSERT the reserved session_id using
-- OVERRIDING SYSTEM VALUE, in its final publication transaction.

CREATE TABLE publication_reservations (
    reservation_id uuid PRIMARY KEY,
    session_id bigint NOT NULL UNIQUE CHECK (session_id >= 0),
    database_id uuid NOT NULL REFERENCES schema_identity(database_id),
    backend_id bigint NOT NULL REFERENCES storage_backends(backend_id),
    created_by bigint NOT NULL REFERENCES users(user_id),
    request jsonb NOT NULL CHECK (jsonb_typeof(request) = 'object'),
    state text NOT NULL DEFAULT 'reserved'
        CHECK (state IN (
            'reserved', 'published', 'cleanup_pending', 'purged'
        )),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE publication_attempts (
    attempt_id uuid PRIMARY KEY,
    reservation_id uuid NOT NULL
        REFERENCES publication_reservations(reservation_id),
    object_prefix text NOT NULL CHECK (pgl_valid_relative_path(object_prefix)),
    state text NOT NULL DEFAULT 'reserved'
        CHECK (state IN (
            'reserved', 'transferring', 'interrupted',
            'published', 'cleanup_pending', 'purged'
        )),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    last_heartbeat_at timestamptz,
    failure_code text,
    UNIQUE (reservation_id, object_prefix)
);

-- At most one authorized/live attempt per reservation.
-- This does not fence filesystem writes from an old process.
CREATE UNIQUE INDEX publication_attempts_one_live
ON publication_attempts (reservation_id)
WHERE state IN ('reserved', 'transferring');

REVOKE ALL PRIVILEGES ON publication_reservations FROM PUBLIC;
REVOKE ALL PRIVILEGES ON publication_attempts FROM PUBLIC;

COMMENT ON TABLE publication_reservations IS
    'Committed ID reservations. Absence of a completed session is not an orphan test.';

COMMENT ON TABLE publication_attempts IS
    'Transfer-attempt accounting. Heartbeat expiry does not authorize deletion.';

-- Before promotion:
-- * add narrow state-transition/identity guards and application grants;
-- * integrate final publication and filesystem ownership/exclusion;
-- * add GC plan/job records and explicit purge authorization;
-- * test on disposable PostgreSQL.

-- Publication draft lifecycle guards v1
-- Publication and GC transitions remain disabled until coordinated
-- implementations exist. No trigger disabling or generic bypass flag.

CREATE FUNCTION pgl_guard_publication_reservation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, pg_temp
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.state <> 'reserved' THEN
            RAISE EXCEPTION 'A reservation must start reserved';
        END IF;
        RETURN NEW;
    END IF;

    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Permanent deletion is not implemented';
    END IF;

    IF OLD.state <> 'reserved'
       OR NEW.state NOT IN ('published', 'superseded')
       OR (to_jsonb(NEW) - 'state')
           IS DISTINCT FROM (to_jsonb(OLD) - 'state') THEN
        RAISE EXCEPTION 'Invalid reservation transition';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM __SCHEMA__.publication_finalization_authority
        WHERE transaction_id = pg_current_xact_id()
          AND reservation_id = OLD.reservation_id
    ) THEN
        RAISE EXCEPTION 'Publication requires authorized finalization';
    END IF;

    IF NEW.state = 'superseded' AND NOT EXISTS (
        SELECT 1
        FROM __SCHEMA__.publication_reuse_results AS result
        JOIN __SCHEMA__.publication_finalization_authority AS auth
          USING (reservation_id, attempt_id)
        WHERE result.reservation_id = OLD.reservation_id
          AND auth.transaction_id = pg_current_xact_id()
    ) THEN
        RAISE EXCEPTION 'Supersession requires a recorded result';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER publication_reservations_guard
BEFORE INSERT OR UPDATE OR DELETE ON publication_reservations
FOR EACH ROW EXECUTE FUNCTION pgl_guard_publication_reservation();

CREATE FUNCTION pgl_guard_publication_attempt()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    owner_state text;
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Permanent deletion is not implemented';
    END IF;

    IF TG_OP = 'UPDATE' THEN
        IF (to_jsonb(NEW) - ARRAY[
                'state', 'last_heartbeat_at', 'failure_code'
            ]::text[])
            IS DISTINCT FROM
           (to_jsonb(OLD) - ARRAY[
                'state', 'last_heartbeat_at', 'failure_code'
            ]::text[]) THEN
            RAISE EXCEPTION 'Attempt identity is immutable';
        END IF;

        IF NEW.state IN ('published', 'superseded') THEN
            IF OLD.state <> 'transferring'
               OR (to_jsonb(NEW) - 'state')
                    IS DISTINCT FROM (to_jsonb(OLD) - 'state')
               OR NOT EXISTS (
                   SELECT 1
                   FROM __SCHEMA__.publication_finalization_authority
                   WHERE transaction_id = pg_current_xact_id()
                     AND reservation_id = OLD.reservation_id
                     AND attempt_id = OLD.attempt_id
               ) THEN
                RAISE EXCEPTION 'Invalid or unauthorized finalization';
            END IF;

            IF NEW.state = 'superseded' AND NOT EXISTS (
                SELECT 1 FROM __SCHEMA__.publication_reuse_results
                WHERE reservation_id = OLD.reservation_id
                  AND attempt_id = OLD.attempt_id
            ) THEN
                RAISE EXCEPTION 'Supersession requires a recorded result';
            END IF;
            RETURN NEW;
        END IF;
    END IF;

    SELECT state INTO owner_state
    FROM __SCHEMA__.publication_reservations
    WHERE reservation_id = NEW.reservation_id FOR UPDATE;

    IF NOT FOUND OR owner_state <> 'reserved' THEN
        RAISE EXCEPTION 'Attempt requires a reserved publication';
    END IF;

    IF TG_OP = 'INSERT' THEN
        IF NEW.state <> 'reserved'
           OR NEW.last_heartbeat_at IS NOT NULL
           OR NEW.failure_code IS NOT NULL
           OR EXISTS (
               SELECT 1 FROM __SCHEMA__.publication_attempts
               WHERE reservation_id = NEW.reservation_id
           ) THEN
            RAISE EXCEPTION 'Only one fresh attempt is supported';
        END IF;
        RETURN NEW;
    END IF;

    IF NOT (
        (OLD.state = 'reserved'
            AND NEW.state IN ('transferring', 'interrupted'))
        OR
        (OLD.state = 'transferring'
            AND NEW.state IN ('transferring', 'interrupted'))
    ) THEN
        RAISE EXCEPTION 'Unsupported attempt transition';
    END IF;

    IF NEW.state = 'transferring' THEN
        IF NEW.failure_code IS NOT NULL THEN
            RAISE EXCEPTION 'Transferring attempt cannot have a failure code';
        END IF;
        NEW.last_heartbeat_at := clock_timestamp();
    ELSE
        IF NEW.last_heartbeat_at IS DISTINCT FROM OLD.last_heartbeat_at THEN
            RAISE EXCEPTION 'Interruption cannot rewrite the last heartbeat';
        END IF;
        IF NEW.failure_code IS NOT NULL
           AND NEW.failure_code !~ '^[a-z][a-z0-9_]{0,63}$' THEN
            RAISE EXCEPTION 'Invalid failure code';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER publication_attempts_guard
BEFORE INSERT OR UPDATE OR DELETE ON publication_attempts
FOR EACH ROW EXECUTE FUNCTION pgl_guard_publication_attempt();


CREATE TABLE __SCHEMA__.publication_writer_credentials (
    attempt_id uuid PRIMARY KEY
        REFERENCES __SCHEMA__.publication_attempts(attempt_id),
    verifier text NOT NULL
        CHECK (verifier ~ '^[0-9a-f]{64}$')
);

REVOKE ALL ON __SCHEMA__.publication_writer_credentials FROM PUBLIC;

CREATE FUNCTION __SCHEMA__.pgl_writer_operation(
    requested_attempt uuid,
    supplied_secret text,
    requested_operation text
)
RETURNS text
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    expected_verifier text;
    actual_verifier text;
    reservation_uuid uuid;
    reservation_state text;
    attempt_state text;
    next_state text;
BEGIN
    -- No supplied secret is included in exception messages.
    IF supplied_secret IS NULL
       OR supplied_secret !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION USING
            ERRCODE = '42501',
            MESSAGE = 'Writer authorization failed';
    END IF;

    SELECT verifier INTO expected_verifier
    FROM __SCHEMA__.publication_writer_credentials
    WHERE attempt_id = requested_attempt;

    -- Match the Python credential_verifier() byte-for-byte.
    actual_verifier := encode(
        sha256(
            convert_to('pgl.publication.writer.v1', 'UTF8')
            || decode('00', 'hex')
            || convert_to(requested_attempt::text, 'UTF8')
            || decode('00', 'hex')
            || decode(supplied_secret, 'hex')
        ),
        'hex'
    );

    IF expected_verifier IS NULL
       OR actual_verifier IS DISTINCT FROM expected_verifier THEN
        RAISE EXCEPTION USING
            ERRCODE = '42501',
            MESSAGE = 'Writer authorization failed';
    END IF;

    IF requested_operation IS NULL
       OR requested_operation NOT IN ('start', 'heartbeat', 'interrupt') THEN
        RAISE EXCEPTION 'Unsupported writer operation';
    END IF;

    SELECT reservation_id INTO reservation_uuid
    FROM __SCHEMA__.publication_attempts
    WHERE attempt_id = requested_attempt;

    -- Use the same lock ordering as the lifecycle guard:
    -- reservation first, attempt second.
    SELECT state INTO reservation_state
    FROM __SCHEMA__.publication_reservations
    WHERE reservation_id = reservation_uuid
    FOR UPDATE;

    SELECT state INTO attempt_state
    FROM __SCHEMA__.publication_attempts
    WHERE attempt_id = requested_attempt
    FOR UPDATE;

    IF reservation_state IS DISTINCT FROM 'reserved' THEN
        RAISE EXCEPTION 'Publication is not available for transfer';
    END IF;

    IF requested_operation = 'start' THEN
        IF attempt_state IS DISTINCT FROM 'reserved' THEN
            RAISE EXCEPTION 'Attempt cannot be started';
        END IF;
        next_state := 'transferring';

    ELSIF requested_operation = 'heartbeat' THEN
        IF attempt_state IS DISTINCT FROM 'transferring' THEN
            RAISE EXCEPTION 'Attempt cannot accept a heartbeat';
        END IF;
        next_state := 'transferring';

    ELSE
        IF attempt_state IS NULL
           OR attempt_state NOT IN ('reserved', 'transferring') THEN
            RAISE EXCEPTION 'Attempt cannot be interrupted';
        END IF;
        next_state := 'interrupted';
    END IF;

    UPDATE __SCHEMA__.publication_attempts
    SET state = next_state,
        failure_code = CASE
            WHEN next_state = 'interrupted' THEN 'writer_interrupted'
            ELSE NULL
        END
    WHERE attempt_id = requested_attempt;

    -- Existing trigger generates heartbeat timestamps and preserves
    -- the last heartbeat when transitioning to interrupted.
    RETURN next_state;
END;
$function$;

-- PostgreSQL grants function execution to PUBLIC by default.
REVOKE ALL ON FUNCTION
    __SCHEMA__.pgl_writer_operation(uuid, text, text)
FROM PUBLIC;


CREATE FUNCTION __SCHEMA__.pgl_reserve_writer(
    supplied_request jsonb,
    supplied_verifier text
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    reservation_uuid uuid;
    attempt_uuid uuid;
    database_uuid uuid;
    requested_backend bigint;
    requested_creator bigint;
    allocated_id bigint;
    stored_request jsonb;
    stored_verifier text;
    destination_prefix text;
    result jsonb;
    field_name text;
BEGIN
    IF jsonb_typeof(supplied_request) IS DISTINCT FROM 'object' THEN
        RAISE EXCEPTION 'Invalid reservation request';
    END IF;

    IF supplied_request->'version' IS DISTINCT FROM '1'::jsonb
       OR jsonb_typeof(supplied_request->'plan') IS DISTINCT FROM 'object'
       OR NOT (supplied_request ?& ARRAY[
           'version', 'database_id', 'reservation_id', 'attempt_id',
           'backend_id', 'created_by', 'plan'
       ])
       OR supplied_request - ARRAY[
           'version', 'database_id', 'reservation_id', 'attempt_id',
           'backend_id', 'created_by', 'plan'
       ]::text[] <> '{}'::jsonb THEN
        RAISE EXCEPTION 'Invalid reservation request';
    END IF;

    FOREACH field_name IN ARRAY ARRAY[
        'database_id', 'reservation_id', 'attempt_id'
    ] LOOP
        IF jsonb_typeof(supplied_request->field_name)
               IS DISTINCT FROM 'string' THEN
            RAISE EXCEPTION 'Invalid reservation identity';
        END IF;

        BEGIN
            IF ((supplied_request->>field_name)::uuid)::text
                   IS DISTINCT FROM supplied_request->>field_name
               OR (supplied_request->>field_name)::uuid =
                   '00000000-0000-0000-0000-000000000000'::uuid THEN
                RAISE EXCEPTION 'Invalid reservation identity';
            END IF;
        EXCEPTION WHEN invalid_text_representation THEN
            RAISE EXCEPTION 'Invalid reservation identity';
        END;
    END LOOP;

    FOREACH field_name IN ARRAY ARRAY['backend_id', 'created_by'] LOOP
        IF jsonb_typeof(supplied_request->field_name)
               IS DISTINCT FROM 'number'
           OR (supplied_request->>field_name) !~ '^(0|[1-9][0-9]*)$' THEN
            RAISE EXCEPTION 'Invalid reservation numeric identity';
        END IF;
    END LOOP;

    IF supplied_verifier IS NULL
       OR supplied_verifier !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'Invalid writer verifier';
    END IF;

    reservation_uuid := (supplied_request->>'reservation_id')::uuid;
    attempt_uuid := (supplied_request->>'attempt_id')::uuid;
    database_uuid := (supplied_request->>'database_id')::uuid;
    requested_backend := (supplied_request->>'backend_id')::bigint;
    requested_creator := (supplied_request->>'created_by')::bigint;

    -- Match the draft Python allocator's retry lock.
    IF NOT pg_try_advisory_xact_lock(hashtextextended(
        'pgl.reserve:' || __SCHEMA_LITERAL__ || ':' ||
        reservation_uuid::text, 0
    )) THEN
        RAISE EXCEPTION USING ERRCODE = '55P03',
            MESSAGE = 'Reservation is currently locked';
    END IF;

    IF (SELECT count(*) FROM __SCHEMA__.schema_identity) <> 1
       OR NOT EXISTS (
           SELECT 1 FROM __SCHEMA__.schema_identity
           WHERE database_id = database_uuid
       ) THEN
        RAISE EXCEPTION 'Database UUID does not match the request';
    END IF;

    SELECT request INTO stored_request
    FROM __SCHEMA__.publication_reservations
    WHERE reservation_id = reservation_uuid;

    IF FOUND THEN
        IF stored_request IS DISTINCT FROM supplied_request THEN
            RAISE EXCEPTION 'Reservation request or writer binding mismatch';
        END IF;

        SELECT verifier INTO stored_verifier
        FROM __SCHEMA__.publication_writer_credentials
        WHERE attempt_id = attempt_uuid;

        IF stored_verifier IS DISTINCT FROM supplied_verifier THEN
            RAISE EXCEPTION 'Reservation request or writer binding mismatch';
        END IF;
    ELSE
        PERFORM 1 FROM __SCHEMA__.storage_backends
        WHERE backend_id = requested_backend AND is_active
        FOR SHARE;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'Storage backend is missing or inactive';
        END IF;

        -- Same sequence used by ordinary session INSERTs.
        allocated_id := nextval(
            pg_get_serial_sequence(
                format('%I.%I', __SCHEMA_LITERAL__, 'session'),
                'session_id'
            )::regclass
        );
        IF allocated_id IS NULL THEN
            RAISE EXCEPTION 'Session allocator is unavailable';
        END IF;

        destination_prefix :=
            'database-' || database_uuid::text ||
            '/session' ||
            CASE WHEN length(allocated_id::text) < 6
                THEN lpad(allocated_id::text, 6, '0')
                ELSE allocated_id::text
            END ||
            '/attempt-' || attempt_uuid::text;

        INSERT INTO __SCHEMA__.publication_reservations (
            reservation_id, session_id, database_id,
            backend_id, created_by, request
        ) VALUES (
            reservation_uuid, allocated_id, database_uuid,
            requested_backend, requested_creator, supplied_request
        );

        INSERT INTO __SCHEMA__.publication_attempts (
            attempt_id, reservation_id, object_prefix
        ) VALUES (
            attempt_uuid, reservation_uuid, destination_prefix
        );

        INSERT INTO __SCHEMA__.publication_writer_credentials (
            attempt_id, verifier
        ) VALUES (attempt_uuid, supplied_verifier);
    END IF;

    SELECT jsonb_build_object(
        'reservation_id', r.reservation_id::text,
        'session_id', r.session_id,
        'reservation_state', r.state,
        'attempt_id', a.attempt_id::text,
        'object_prefix', a.object_prefix,
        'attempt_state', a.state
    ) INTO result
    FROM __SCHEMA__.publication_reservations AS r
    JOIN __SCHEMA__.publication_attempts AS a USING (reservation_id)
    WHERE r.reservation_id = reservation_uuid
      AND a.attempt_id = attempt_uuid;

    IF result IS NULL THEN
        RAISE EXCEPTION 'Reservation has no matching attempt';
    END IF;

    RETURN result;
END;
$function$;

REVOKE ALL ON FUNCTION
    __SCHEMA__.pgl_reserve_writer(jsonb, text)
FROM PUBLIC;


-- Deferred consistency guard: checked at transaction commit.
-- No publication or GC privileges are granted here.
--
-- Only reserved checkpoints are covered. Existing unreserved saves
-- are not silently enrolled in this new protocol.

CREATE FUNCTION __SCHEMA__.pgl_check_publication_atomicity()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    reserved_id uuid;
    reservation_state text;
    checkpoint_state text;
    attempt_count bigint;
    published_count bigint;
BEGIN
    IF TG_TABLE_NAME = 'session' THEN
        SELECT reservation_id INTO reserved_id
        FROM __SCHEMA__.publication_reservations
        WHERE session_id = NEW.session_id;
    ELSE
        reserved_id := NEW.reservation_id;
    END IF;

    -- Ordinary unreserved checkpoints are outside this draft protocol.
    IF reserved_id IS NULL THEN
        RETURN NULL;
    END IF;

    -- Read current transaction-final state, not the historical NEW
    -- values from when this deferred event was queued.
    SELECT r.state, s.status
    INTO reservation_state, checkpoint_state
    FROM __SCHEMA__.publication_reservations AS r
    LEFT JOIN __SCHEMA__.session AS s
        ON s.session_id = r.session_id
    WHERE r.reservation_id = reserved_id;

    SELECT count(*), count(*) FILTER (WHERE state = 'published')
    INTO attempt_count, published_count
    FROM __SCHEMA__.publication_attempts
    WHERE reservation_id = reserved_id;

    IF reservation_state = 'published'
       OR checkpoint_state = 'complete'
       OR published_count > 0 THEN

        IF reservation_state IS DISTINCT FROM 'published'
           OR checkpoint_state IS DISTINCT FROM 'complete'
           OR attempt_count <> 1
           OR published_count <> 1 THEN
            RAISE EXCEPTION USING
                ERRCODE = '23514',
                MESSAGE =
                    'Checkpoint, reservation, and attempt publication '
                    'must commit together';
        END IF;
    END IF;

    RETURN NULL;
END;
$function$;

REVOKE ALL ON FUNCTION
    __SCHEMA__.pgl_check_publication_atomicity()
FROM PUBLIC;

CREATE CONSTRAINT TRIGGER session_publication_atomicity
AFTER INSERT OR UPDATE ON __SCHEMA__.session
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW
EXECUTE FUNCTION __SCHEMA__.pgl_check_publication_atomicity();

CREATE CONSTRAINT TRIGGER reservation_publication_atomicity
AFTER INSERT OR UPDATE ON __SCHEMA__.publication_reservations
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW
EXECUTE FUNCTION __SCHEMA__.pgl_check_publication_atomicity();

CREATE CONSTRAINT TRIGGER attempt_publication_atomicity
AFTER INSERT OR UPDATE ON __SCHEMA__.publication_attempts
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW
EXECUTE FUNCTION __SCHEMA__.pgl_check_publication_atomicity();


-- Private authorization records exist only during the finalizer's
-- statement. Applications receive no privileges on this table.
CREATE TABLE __SCHEMA__.publication_finalization_authority (
    transaction_id xid8 NOT NULL,
    reservation_id uuid NOT NULL,
    attempt_id uuid NOT NULL,
    PRIMARY KEY (transaction_id, reservation_id, attempt_id)
);
REVOKE ALL ON __SCHEMA__.publication_finalization_authority FROM PUBLIC;

-- Extend the existing guards with narrowly scoped publication branches.
-- Preserve their original checks for every other operation.


CREATE FUNCTION __SCHEMA__.pgl_finalize_writer(
    requested_attempt uuid,
    supplied_secret text,
    expected_manifest_hash text
)
RETURNS bigint
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    expected_verifier text;
    actual_verifier text;
    reservation_uuid uuid;
    reserved_session bigint;
    reservation_state text;
    attempt_state text;
    checkpoint_state text;
    checkpoint_hash text;
    checkpoint_deleted timestamptz;
BEGIN
    IF supplied_secret IS NULL
       OR supplied_secret !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION USING
            ERRCODE = '42501',
            MESSAGE = 'Writer authorization failed';
    END IF;

    SELECT verifier INTO expected_verifier
    FROM __SCHEMA__.publication_writer_credentials
    WHERE attempt_id = requested_attempt;

    actual_verifier := encode(
        sha256(
            convert_to('pgl.publication.writer.v1', 'UTF8')
            || decode('00', 'hex')
            || convert_to(requested_attempt::text, 'UTF8')
            || decode('00', 'hex')
            || decode(supplied_secret, 'hex')
        ),
        'hex'
    );

    IF expected_verifier IS NULL
       OR actual_verifier IS DISTINCT FROM expected_verifier THEN
        RAISE EXCEPTION USING
            ERRCODE = '42501',
            MESSAGE = 'Writer authorization failed';
    END IF;

    IF expected_manifest_hash IS NULL
       OR expected_manifest_hash !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'Invalid expected manifest identity';
    END IF;

    SELECT reservation_id INTO reservation_uuid
    FROM __SCHEMA__.publication_attempts
    WHERE attempt_id = requested_attempt;

    SELECT session_id, state
    INTO reserved_session, reservation_state
    FROM __SCHEMA__.publication_reservations
    WHERE reservation_id = reservation_uuid
    FOR UPDATE;

    SELECT state INTO attempt_state
    FROM __SCHEMA__.publication_attempts
    WHERE attempt_id = requested_attempt
    FOR UPDATE;

    SELECT status, manifest_hash::text, deleted_at
    INTO checkpoint_state, checkpoint_hash, checkpoint_deleted
    FROM __SCHEMA__.session
    WHERE session_id = reserved_session
    FOR UPDATE;

    IF checkpoint_state IS DISTINCT FROM 'complete'
       OR checkpoint_hash IS DISTINCT FROM expected_manifest_hash
       OR checkpoint_deleted IS NOT NULL THEN
        RAISE EXCEPTION
            'Finalization requires the matching active completed checkpoint';
    END IF;

    IF reservation_state = 'published' AND attempt_state = 'published' THEN
        RETURN reserved_session;
    END IF;

    IF reservation_state IS DISTINCT FROM 'reserved'
       OR attempt_state IS DISTINCT FROM 'transferring' THEN
        RAISE EXCEPTION 'Attempt is not eligible for finalization';
    END IF;

    INSERT INTO __SCHEMA__.publication_finalization_authority (
        transaction_id, reservation_id, attempt_id
    ) VALUES (
        pg_current_xact_id(), reservation_uuid, requested_attempt
    );

    UPDATE __SCHEMA__.publication_attempts
    SET state = 'published'
    WHERE attempt_id = requested_attempt;

    UPDATE __SCHEMA__.publication_reservations
    SET state = 'published'
    WHERE reservation_id = reservation_uuid;

    DELETE FROM __SCHEMA__.publication_finalization_authority
    WHERE transaction_id = pg_current_xact_id()
      AND reservation_id = reservation_uuid
      AND attempt_id = requested_attempt;

    RETURN reserved_session;
END;
$function$;

REVOKE ALL ON FUNCTION
    __SCHEMA__.pgl_finalize_writer(uuid, text, text)
FROM PUBLIC;


ALTER TABLE __SCHEMA__.publication_reservations
    DROP CONSTRAINT publication_reservations_state_check,
    ADD CONSTRAINT publication_reservations_state_check
    CHECK (state IN (
        'reserved', 'published', 'superseded', 'cleanup_pending', 'purged'
    ));

ALTER TABLE __SCHEMA__.publication_attempts
    DROP CONSTRAINT publication_attempts_state_check,
    ADD CONSTRAINT publication_attempts_state_check
    CHECK (state IN (
        'reserved', 'transferring', 'interrupted', 'published',
        'superseded', 'cleanup_pending', 'purged'
    ));

-- Result IDs are historical, database-local identifiers, deliberately
-- not foreign keys to session. Purging the target must remain possible.
CREATE TABLE __SCHEMA__.publication_reuse_results (
    reservation_id uuid PRIMARY KEY
        REFERENCES __SCHEMA__.publication_reservations(reservation_id),
    attempt_id uuid NOT NULL UNIQUE
        REFERENCES __SCHEMA__.publication_attempts(attempt_id),
    result_session_id bigint NOT NULL CHECK (result_session_id >= 0),
    manifest_hash text NOT NULL CHECK (manifest_hash ~ '^[0-9a-f]{64}$'),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
REVOKE ALL ON __SCHEMA__.publication_reuse_results FROM PUBLIC;

CREATE FUNCTION __SCHEMA__.pgl_guard_reuse_result()
RETURNS trigger LANGUAGE plpgsql AS $function$
BEGIN
    RAISE EXCEPTION 'Recorded publication reuse results are immutable';
END;
$function$;

CREATE TRIGGER publication_reuse_results_immutable
BEFORE UPDATE OR DELETE ON __SCHEMA__.publication_reuse_results
FOR EACH ROW EXECUTE FUNCTION __SCHEMA__.pgl_guard_reuse_result();

-- Prototype installation mechanism, consistent with the existing
-- finalization draft. Replace with explicit guard definitions before
-- promoting the consolidated migration.


CREATE FUNCTION __SCHEMA__.pgl_check_reuse_atomicity()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    reservation_state text;
    attempt_state text;
    recorded_attempt uuid;
    owned_attempt uuid;
    reserved_session bigint;
    target_session bigint;
    attempt_count bigint;
BEGIN
    SELECT r.state, r.session_id, x.attempt_id, x.result_session_id
    INTO reservation_state, reserved_session, recorded_attempt, target_session
    FROM __SCHEMA__.publication_reservations AS r
    LEFT JOIN __SCHEMA__.publication_reuse_results AS x
      USING (reservation_id)
    WHERE r.reservation_id = NEW.reservation_id;

    SELECT count(*) INTO attempt_count
    FROM __SCHEMA__.publication_attempts
    WHERE reservation_id = NEW.reservation_id;

    SELECT attempt_id, state INTO owned_attempt, attempt_state
    FROM __SCHEMA__.publication_attempts
    WHERE reservation_id = NEW.reservation_id;

    IF reservation_state = 'superseded'
       OR attempt_state = 'superseded'
       OR recorded_attempt IS NOT NULL THEN
        IF reservation_state IS DISTINCT FROM 'superseded'
           OR attempt_state IS DISTINCT FROM 'superseded'
           OR attempt_count <> 1
           OR recorded_attempt IS NULL
           OR recorded_attempt IS DISTINCT FROM owned_attempt
           OR target_session = reserved_session
           OR EXISTS (
               SELECT 1 FROM __SCHEMA__.session
               WHERE session_id = reserved_session
           ) THEN
            RAISE EXCEPTION 'Reuse result and supersession must commit together';
        END IF;
    END IF;
    RETURN NULL;
END;
$function$;

REVOKE ALL ON FUNCTION __SCHEMA__.pgl_check_reuse_atomicity() FROM PUBLIC;

CREATE CONSTRAINT TRIGGER reservation_reuse_atomicity
AFTER INSERT OR UPDATE ON __SCHEMA__.publication_reservations
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION __SCHEMA__.pgl_check_reuse_atomicity();

CREATE CONSTRAINT TRIGGER attempt_reuse_atomicity
AFTER INSERT OR UPDATE ON __SCHEMA__.publication_attempts
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION __SCHEMA__.pgl_check_reuse_atomicity();

CREATE CONSTRAINT TRIGGER result_reuse_atomicity
AFTER INSERT ON __SCHEMA__.publication_reuse_results
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION __SCHEMA__.pgl_check_reuse_atomicity();

CREATE FUNCTION __SCHEMA__.pgl_supersede_writer(
    requested_attempt uuid,
    supplied_secret text,
    requested_result bigint,
    expected_hash text
)
RETURNS bigint
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    stored_verifier text;
    actual_verifier text;
    reservation_uuid uuid;
    reserved_session bigint;
    reservation_state text;
    attempt_state text;
    recorded_result bigint;
    recorded_hash text;
BEGIN
    IF supplied_secret IS NULL OR supplied_secret !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION USING ERRCODE = '42501',
            MESSAGE = 'Writer authorization failed';
    END IF;

    SELECT verifier INTO stored_verifier
    FROM __SCHEMA__.publication_writer_credentials
    WHERE attempt_id = requested_attempt;

    actual_verifier := encode(sha256(
        convert_to('pgl.publication.writer.v1', 'UTF8')
        || decode('00', 'hex')
        || convert_to(requested_attempt::text, 'UTF8')
        || decode('00', 'hex')
        || decode(supplied_secret, 'hex')
    ), 'hex');

    IF stored_verifier IS NULL
       OR actual_verifier IS DISTINCT FROM stored_verifier THEN
        RAISE EXCEPTION USING ERRCODE = '42501',
            MESSAGE = 'Writer authorization failed';
    END IF;

    IF requested_result IS NULL OR requested_result < 0
       OR expected_hash IS NULL OR expected_hash !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'Invalid reuse result identity';
    END IF;

    SELECT reservation_id INTO reservation_uuid
    FROM __SCHEMA__.publication_attempts
    WHERE attempt_id = requested_attempt;

    SELECT session_id, state INTO reserved_session, reservation_state
    FROM __SCHEMA__.publication_reservations
    WHERE reservation_id = reservation_uuid FOR UPDATE;

    SELECT state INTO attempt_state
    FROM __SCHEMA__.publication_attempts
    WHERE attempt_id = requested_attempt FOR UPDATE;

    IF requested_result = reserved_session
       OR EXISTS (
           SELECT 1 FROM __SCHEMA__.session
           WHERE session_id = reserved_session
       ) THEN
        RAISE EXCEPTION 'Reserved checkpoint must remain unpublished for reuse';
    END IF;

    -- Keep the result active and stable through this transaction.
    PERFORM 1 FROM __SCHEMA__.session
    WHERE session_id = requested_result
      AND origin = 'raw'
      AND status = 'complete'
      AND deleted_at IS NULL
      AND contract_version = 'pgl.raw.session.v2'
      AND manifest_hash = expected_hash
    FOR SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Reuse result is missing, deleted, incomplete, or does not match';
    END IF;

    SELECT result_session_id, manifest_hash
    INTO recorded_result, recorded_hash
    FROM __SCHEMA__.publication_reuse_results
    WHERE reservation_id = reservation_uuid;

    IF FOUND THEN
        IF reservation_state IS DISTINCT FROM 'superseded'
           OR attempt_state IS DISTINCT FROM 'superseded'
           OR recorded_result IS DISTINCT FROM requested_result
           OR recorded_hash IS DISTINCT FROM expected_hash THEN
            RAISE EXCEPTION 'Recorded reuse result does not match this retry';
        END IF;
        RETURN recorded_result;
    END IF;

    IF reservation_state IS DISTINCT FROM 'reserved'
       OR attempt_state IS DISTINCT FROM 'transferring' THEN
        RAISE EXCEPTION 'Attempt is not eligible for supersession';
    END IF;

    INSERT INTO __SCHEMA__.publication_reuse_results (
        reservation_id, attempt_id, result_session_id, manifest_hash
    ) VALUES (
        reservation_uuid, requested_attempt, requested_result, expected_hash
    );

    INSERT INTO __SCHEMA__.publication_finalization_authority
        (transaction_id, reservation_id, attempt_id)
    VALUES (pg_current_xact_id(), reservation_uuid, requested_attempt);

    UPDATE __SCHEMA__.publication_attempts SET state = 'superseded'
    WHERE attempt_id = requested_attempt;

    UPDATE __SCHEMA__.publication_reservations SET state = 'superseded'
    WHERE reservation_id = reservation_uuid;

    DELETE FROM __SCHEMA__.publication_finalization_authority
    WHERE transaction_id = pg_current_xact_id()
      AND reservation_id = reservation_uuid
      AND attempt_id = requested_attempt;

    RETURN requested_result;
END;
$function$;

REVOKE ALL ON FUNCTION
    __SCHEMA__.pgl_supersede_writer(uuid, text, bigint, text)
FROM PUBLIC;



CREATE FUNCTION __SCHEMA__.pgl_guard_writer_credential()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, pg_temp
AS $function$
BEGIN
    RAISE EXCEPTION 'Writer bindings are immutable';
END;
$function$;

CREATE TRIGGER publication_writer_credentials_immutable
BEFORE UPDATE OR DELETE ON __SCHEMA__.publication_writer_credentials
FOR EACH ROW EXECUTE FUNCTION __SCHEMA__.pgl_guard_writer_credential();

$pgl_publication_definition$;
BEGIN
    IF target_schema IS NULL OR target_schema = 'pg_catalog' THEN
        RAISE EXCEPTION 'Application schema must be selected';
    END IF;
    definition := replace(definition, '__SCHEMA_LITERAL__', quote_literal(target_schema));
    definition := replace(definition, '__SCHEMA__', quote_ident(target_schema));
    EXECUTE definition;
END;
$pgl_install_publication$;
