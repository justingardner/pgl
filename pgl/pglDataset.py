################################################################
#   filename: pglDataset.py
#    purpose: Provides dataset class for managing dataset membership and metadata.
#         by: JLG
#       date: Oct 18, 2026
################################################################ 
"""Dataset membership and lightweight checkpoint metadata.

Session payloads remain owned by their saved session checkpoints.
Dataset scientific-product persistence is added separately.
"""

from copy import deepcopy
import json
import math

from traitlets import Dict, Int, List, Unicode

from .pglMessages import pglMessages
from .pglPipeline import pglActionable


class pglDataset(pglActionable):
    """A working dataset referencing exact saved session checkpoints.

    Membership is independent of cached working sessions.
    Editing or saving a cached session never repoints membership.

    Persistence and database access are delegated to pglStorage.
    """

    datasetID = Int(default_value=None, allow_none=True, help="Most recently loaded or saved dataset checkpoint ID").tag(serialize=False)
    checkpointName = Unicode(default_value=None, allow_none=True, help="Editable checkpoint label").tag(serialize=False)
    description = Unicode(default_value=None, allow_none=True, help="Editable checkpoint description").tag(serialize=False)

    derivedParameters = Dict(default_value={}, help="Small derived results stored as strict PGL metadata")
    data = Dict(default_value={}, help="Attached scientific products; persisted separately from metadata").tag(serialize=False)

    _members = List(default_value=[], help="Ordered exact checkpoint membership").tag(serialize=False)
    _sessionCache = Dict(default_value={}, help="Working session objects cached by original member ID").tag(serialize=False)
    _datasetDatabaseIdentity = Dict(default_value={}, help="Bound runtime database endpoint").tag(serialize=False)
    _datasetBaseline = Dict(default_value={}, help="Storage-managed immutable history boundary").tag(serialize=False)

    def __init__(self, name=None, *, settings=None, settingsName=None):
        """Create a working dataset without connecting or loading sessions.

        Empty construction is also used by strict metadata restoration.
        Database binding occurs when membership is resolved or saved.
        """
        super().__init__()

        if settings is not None and settingsName is not None:
            raise ValueError("Supply settings or settingsName, not both.")

        if name is not None:
            self.name = self._validateName(name)

        # Runtime configuration is deliberately excluded from serialization.
        self._datasetSettings = settings
        self._datasetSettingsName = settingsName
        self._datasetLoadOptions = {}

    @staticmethod
    def _validateName(value):
        """Require a meaningful name without silently normalizing it."""
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError("Dataset name must be a nonblank string without NUL.")

        return value

    @staticmethod
    def _validateSessionID(value):
        """Validate an exact database checkpoint ID, excluding booleans."""
        if type(value) is not int or not 0 <= value <= 9223372036854775807:
            raise ValueError("sessionID must be a nonnegative PostgreSQL bigint.")

        return value

    @staticmethod
    def _validateEntryName(value):
        """Preserve case and whitespace in supplied nonblank aliases."""
        if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
            raise ValueError("entryName must be a nonblank string without NUL, or None.")

        return value

    @staticmethod
    def _validateLabels(value):
        """Copy structured labels with a portable, plain-JSON representation.

        Labels permit string-keyed objects, lists, finite numbers, booleans,
        strings, and None. Scientific arrays belong in data products.
        """
        if value is None:
            value = {}

        if not isinstance(value, dict):
            raise TypeError("labels must be a dictionary or None.")

        active = set()

        def check(item, path):
            if item is None or type(item) in (bool, int):
                return

            if isinstance(item, str):
                if "\x00" in item:
                    raise ValueError(f"{path}: strings must not contain NUL.")
                return

            if type(item) is float:
                if not math.isfinite(item):
                    raise ValueError(f"{path}: numbers must be finite.")
                return

            if not isinstance(item, (dict, list)):
                raise TypeError(f"{path}: unsupported label value {type(item).__name__}.")

            identity = id(item)

            if identity in active:
                raise ValueError(f"{path}: circular labels are not supported.")

            active.add(identity)

            try:
                if isinstance(item, dict):
                    for key, child in item.items():
                        if not isinstance(key, str) or "\x00" in key:
                            raise ValueError(f"{path}: label keys must be strings without NUL.")

                        check(child, f"{path}.{key}")
                else:
                    for index, child in enumerate(item):
                        check(child, f"{path}[{index}]")
            finally:
                active.remove(identity)

        check(value, "labels")

        # This also returns an independent copy with ordinary JSON types.
        return json.loads(json.dumps(value, allow_nan=False))

    @classmethod
    def _validateMembers(cls, records):
        """Validate and copy the authoritative ordered membership list."""
        if not isinstance(records, list):
            raise TypeError("Dataset membership must be a list.")

        sessionIDs = set()
        entryNames = set()
        result = []

        for record in records:
            if not isinstance(record, dict) or set(record) != {"sessionID", "entryName", "labels"}:
                raise ValueError("Invalid dataset membership fields.")

            sessionID = cls._validateSessionID(record["sessionID"])
            entryName = cls._validateEntryName(record["entryName"])
            labels = cls._validateLabels(record["labels"])

            if sessionID in sessionIDs:
                raise ValueError(f"Duplicate member checkpoint: {sessionID}")

            if entryName is not None and entryName in entryNames:
                raise ValueError(f"Duplicate entryName: {entryName!r}")

            sessionIDs.add(sessionID)

            if entryName is not None:
                entryNames.add(entryName)

            result.append({"sessionID": sessionID, "entryName": entryName, "labels": labels})

        return result

    @property
    def members(self):
        """Return an independent membership snapshot, not a mutable registry."""
        return deepcopy(self._members)

    def _findMember(self, *, sessionID=None, entryName=None):
        """Resolve exactly one selector without reading a session."""
        if (sessionID is None) == (entryName is None):
            raise ValueError("Supply exactly one of sessionID or entryName.")

        if sessionID is not None:
            self._validateSessionID(sessionID)
        else:
            self._validateEntryName(entryName)

        for record in self._members:
            if sessionID is not None and record["sessionID"] == sessionID:
                return record

            if entryName is not None and record["entryName"] == entryName:
                return record

        selector = f"sessionID={sessionID}" if sessionID is not None else f"entryName={entryName!r}"
        raise KeyError(f"No dataset member with {selector}.")

    def addSession(self, sessionID, entryName=None, labels=None, *, settings=None, settingsName=None, sslRootCert=None):
        """Add one or several active completed checkpoints.

        sessionID:
            One checkpoint ID, or a list/tuple of checkpoint IDs.

        Existing members and repeated IDs in the supplied collection warn
        and are skipped. Existing aliases and labels remain unchanged.

        New members preserve their supplied order.
        labels applies independently to each new member.
        entryName is permitted only when exactly one ID is supplied.

        Validates all new members before changing membership. If any new
        checkpoint is missing, deleted, or incomplete, none are added.

        Reads database metadata only. Does not load session payloads,
        populate the session cache, or publish a dataset checkpoint.

        Returns True if members were added, otherwise False.
        Invalid arguments and unavailable source checkpoints raise.
        Configuration/connection cancellation returns False.
        """
        from .pglStorage import pglStorage

        requestedIDs = list(sessionID) if isinstance(sessionID, (list, tuple)) else [sessionID]

        # Validate arguments before database access or membership changes.
        for requestedID in requestedIDs:
            self._validateSessionID(requestedID)

        self._validateEntryName(entryName)
        normalizedLabels = self._validateLabels(labels)

        if entryName is not None and len(requestedIDs) != 1:
            raise ValueError("entryName requires exactly one session ID. Add separately named members with individual addSession() calls.")

        currentMembers = self.members
        existingIDs = {record["sessionID"] for record in currentMembers}
        seenIDs = set()
        newIDs = []
        existingDuplicates = []
        repeatedIDs = []

        for requestedID in requestedIDs:
            if requestedID in seenIDs:
                if requestedID not in repeatedIDs:
                    repeatedIDs.append(requestedID)
                continue

            seenIDs.add(requestedID)

            if requestedID in existingIDs:
                existingDuplicates.append(requestedID)
            else:
                newIDs.append(requestedID)

        if existingDuplicates:
            listed = ", ".join(str(value) for value in existingDuplicates)
            pglMessages.warning(f"Session checkpoint(s) already in this dataset; skipped: {listed}. Existing entry names and labels were not changed.", level=0)

        if repeatedIDs:
            listed = ", ".join(str(value) for value in repeatedIDs)
            pglMessages.warning(f"Repeated session checkpoint ID(s) in the supplied list; duplicate occurrences skipped: {listed}.", level=0)

        if not newIDs:
            return False

        newMembers = [
            {"sessionID": requestedID, "entryName": entryName, "labels": normalizedLabels}
            for requestedID in newIDs
        ]

        # Validates alias uniqueness and creates independent label copies.
        proposed = self._validateMembers([*currentMembers, *newMembers])

        # Resolve all candidate checkpoints in one metadata query.
        summaries = pglStorage.getDatasetMemberSummaries(self, sessionIDs=newIDs, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert)

        if summaries is None:
            return False

        problems = []

        for requestedID in newIDs:
            summary = summaries.get(requestedID)

            if summary is None:
                problems.append(f"{requestedID}: unknown checkpoint")
            elif summary["deletedAt"] is not None:
                problems.append(f"{requestedID}: deleted checkpoint")
            elif summary["status"] != "complete":
                problems.append(f"{requestedID}: status={summary['status']!r}")

        if problems:
            raise ValueError("No members were added. Every new member must be complete and active: " + "; ".join(problems))

        if self.members != currentMembers:
            raise RuntimeError("Dataset membership changed during validation. No members were added by this call; retry with the current membership.")

        # One assignment after every new checkpoint has passed validation.
        # Existing behavior products remain unchanged construction snapshots.
        self._members = proposed
        return True

    def removeSession(self, sessionID=None, *, entryName=None):
        """Remove working membership without changing saved dataset versions."""
        member = self._findMember(sessionID=sessionID, entryName=entryName)
        originalID = member["sessionID"]

        self._members = [record for record in self._members if record["sessionID"] != originalID]
        self._sessionCache.pop(originalID, None)
        return True

    def setEntryMetadata(self, sessionID=None, *, entryName=None, labels=None):
        """Replace a member's alias and labels in this working dataset.

        Supply both desired values. None clears the alias or labels.
        Does not edit the referenced session or any saved dataset version.
        """
        self._validateSessionID(sessionID)
        self._findMember(sessionID=sessionID)

        records = self.members

        for record in records:
            if record["sessionID"] == sessionID:
                record["entryName"] = entryName
                record["labels"] = labels
                break

        self._members = self._validateMembers(records)

    def getSession(self, sessionID=None, *, entryName=None, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None, cacheDirectory=None, directLocalPolicy=None):
        """Return the cached working session, loading it on first access.

        The cache key remains the original member checkpoint ID even if the
        returned session is subsequently saved as a different checkpoint.
        """
        from .pglStorage import pglStorage

        member = self._findMember(sessionID=sessionID, entryName=entryName)

        return pglStorage.getDatasetSession(self, member["sessionID"], settings=settings, settingsName=settingsName, storageOptionsByBackend=storageOptionsByBackend, sslRootCert=sslRootCert, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy)

    def print(self, verbose=False, *, settings=None, settingsName=None, sslRootCert=None):
        """Print member metadata without constructing sessions or matrices."""
        from .pglStorage import pglStorage

        return pglStorage.printDataset(self, verbose=verbose, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert)

    def save(self, *, settings=None, settingsName=None, backendID=None, storageOptions=None, sslRootCert=None):
        """Publish a new dataset checkpoint, including attached products.

        Saved references reuse their registered payloads.
        Attached mutable matrices publish new revisions.
        Does not construct missing products or rerun analyses.
        """
        from .pglStorage import pglStorage

        return pglStorage.saveDataset(self, settings=settings, settingsName=settingsName, backendID=backendID, storageOptions=storageOptions, sslRootCert=sslRootCert)

    def getData(self, productName, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None):
        """Return an attached product, opening a saved reference on demand.

        The opened matrix replaces its reference in dataset.data and is cached.
        Subsequent saves conservatively publish that mutable matrix anew.

        This persistence-stage implementation does not yet extract missing
        products or evaluate input freshness.
        """
        from .pglStorage import pglStorage

        return pglStorage.getDatasetData(self, productName, settings=settings, settingsName=settingsName, storageOptionsByBackend=storageOptionsByBackend, sslRootCert=sslRootCert)
    
    @classmethod
    def load(cls, datasetID, *, settings=None, settingsName=None, sslRootCert=None, includeDeleted=False, storageOptionsByBackend=None, cacheDirectory=None, directLocalPolicy="materialize"):
        """Restore membership and metadata without loading member sessions."""
        from .pglStorage import pglStorage

        return pglStorage.loadDataset(datasetID, settings=settings, settingsName=settingsName, sslRootCert=sslRootCert, includeDeleted=includeDeleted, storageOptionsByBackend=storageOptionsByBackend, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy)
    
    def getBehaviorData(self, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None, cacheDirectory=None, directLocalPolicy=None):
        """Return the dataset-owned comprehensive behavioral matrix.

        Reuses data["behavior"] when present, resolving a saved reference
        through the normal product loader and validating its behavioral schema.

        Otherwise constructs from exact saved member checkpoints, in membership
        order, without using editable sessions in _sessionCache.

        Successful construction attaches a working pglDataMatrix.
        Does not automatically rebuild after membership or code changes.
        Existing products remain construction snapshots.

        Reads sources as needed; does not publish.
        Use save() to persist the attached product explicitly.
        """
        from .pglStorage import pglStorage

        return pglStorage.getDatasetBehaviorData(self, settings=settings, settingsName=settingsName, storageOptionsByBackend=storageOptionsByBackend, sslRootCert=sslRootCert, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy)
    
    def getBehaviorTable(self, taskName=None, *, settings=None, settingsName=None, storageOptionsByBackend=None, sslRootCert=None, cacheDirectory=None, directLocalPolicy=None, **options):
        """Return dataset behavioral trials as an independent pandas DataFrame.

        Obtains the dataset-owned comprehensive matrix through getBehaviorData().

        If data["behavior"] exists:
            Opens a saved product reference when necessary, validates the
            behavioral schema, and reuses the resulting matrix.

        Otherwise:
            Constructs behavior from exact saved member checkpoints in
            membership order. Does not use editable sessions in _sessionCache.
            Attaches the comprehensive matrix only after successful construction.

        Users do not need to call getBehaviorData() first.

        Returns None with a pglMessages.warning for invalid table requests,
        missing/ambiguous task selections, unavailable requested columns,
        and unimplemented projection features. Configuration/connection
        cancellation may also return None.

        Source-construction failures, malformed scientific data, and unexpected
        implementation errors still propagate.

        Each row represents a selected recorded trial START. Unanswered and
        incomplete trials are retained. The DataFrame index preserves combined
        source-matrix trial IDs; sourceTrialNum is the recorded task-local
        trial number, not a dataset-wide identifier.

        Dataset source identity
        -----------------------
        Combined matrices preserve:
            sessionIndex:
                Zero-based member position at construction.
            sourceCheckpointID:
                Referenced member checkpoint ID, scoped to the source database.
            entryName:
                Dataset-local alias captured at construction.
            sourceRunIndex:
                Original run index within the member session.
            sourceTrialIndex:
                Original trial coordinate within the member's extracted matrix.

        runIndex is unique across the combined dataset.
        taskIndex remains local to its source run.
        These fields describe the product's construction snapshot, not
        necessarily the dataset's current working membership.

        Task selection
        --------------
        taskName : str, optional
            Exact match against recorded taskSaveName, not the human-readable
            taskName. Example: "memoryDescriptionTask".
        taskIndex : int, optional
            Zero-based task index, resolved independently within every run
            of every member represented in the matrix.
        taskSelection : "all", optional
            Include trials from all tasks.

        Specify at most one task-selection option. Without one, every run
        must contain exactly one task. Missing or ambiguous matches produce
        a warning and return None; no partial table is returned.
        Automatic main-task resolution is not yet implemented.

        A requested task missing from even one run does not cause that run
        to be silently skipped. Repeated taskSaveName values within a run
        require an unambiguous taskIndex selection.

        Response summaries
        ------------------
        responsePolicy : {"first", "last"}, default "first"
            Select the summary response by RECORDED EVENT ORDER, not sorted
            timestamp order. responseCount counts all accepted response
            events associated with the trial, not physical keypresses.
        includeAllResponses : bool, default False
            Add response1, responseType1, responseTime1, etc., while retaining
            the ordinary summary columns. All times use the same reference.

        Response timing
        ---------------
        responseReference : {"trial", "segment", "trigger"}, default "trial"
            responseTime = response timestamp - reference timestamp, seconds.

            "trial":
                Use the recorded trial START timestamp.
            "segment":
                Require exactly one associated task-scoped segment boundary.
            "trigger":
                Require exactly one eligible associated task-scoped event
                with the requested recorded trigger name.

        segmentNum : int, required for segment timing
            Recorder's ZERO-BASED segment number. segmentNum=1 means the
            second segment; segmentNum=2 means the third.
        segmentBoundary : {"start", "end"}, default "start"
            Use an actually recorded boundary. Segment END events are not
            inferred from the next boundary.
        triggerName : str, required for trigger timing
            Exact recorded trigger name.
        requireTriggerSent : bool, default True
            Trigger timing requires recorded triggerSent=True.
            With False, failed or unspecified attempts also become eligible;
            multiple eligible attempts make the reference ambiguous.

        segmentNum and a nondefault segmentBoundary require segment timing.
        triggerName and requireTriggerSent=False require trigger timing.
        Invalid combinations warn and return None.

        Missing or ambiguous recorded references produce missing responseTime,
        not missing responses. The table is still returned with diagnostics.
        Nonfinite/missing timestamps also produce missing latency.
        Negative offsets are preserved and diagnosed, never clipped.

        Recorded order and timestamp order can differ. A segment START is
        not automatically verified stimulus onset. Timestamp comparisons are
        restricted to the associated trial; clocks are not assumed comparable
        across independent runs or members.

        Run-scoped temporal trigger association is not performed.

        Timing details are available in:
            trials.attrs["viewSpecification"]
            trials.attrs["responseTimingDiagnostics"]
            trials.attrs["responseTimingReferences"]

        Display and column selection
        ----------------------------
        includeAllColumns : bool, default False
            Show internal indices, taskSaveName, and unsupported field markers.
            Also enables numbered response expansion, even if
            includeAllResponses=False was supplied.

            Defaults hide runIndex, taskIndex, taskSaveName,
            trialStartEventIndex, responseEventIndex and numbered variants,
            and fields marked extractionStatus="unsupported".

            Dataset identity fields such as sessionIndex, sourceCheckpointID,
            entryName, sourceRunIndex, and sourceTrialIndex are not hidden by
            that default policy.

            This means all available TRIAL TABLE columns, not raw event rows,
            configuration export, or unimplemented trigger expansion.
        columns : list[str] or tuple[str], optional
            Return exactly these columns in the requested order. Explicit
            selection can expose otherwise hidden or absent parameter fields.
            Unknown or duplicate column names warn and return None.
            Requesting numbered response columns enables response expansion.
            Numbered columns exist only up to the maximum recorded response
            count among the selected trials.

        Automatically returned parameters must be present in at least one
        selected trial. Parameters belonging only to unselected tasks are
        omitted, even with includeAllColumns=True.

        Fields absent from some members use the combined matrix's presence
        masks. Current encoding treats recorded None and absence alike;
        recorded floating NaNs remain present.

        Fixed-length vectors become zero-based ComponentN columns. Missing
        integer/Boolean/string values use pandas nullable representations.
        Absent floating values appear as NaN; ordinary pandas float columns
        do not distinguish them from recorded NaNs. The comprehensive matrix's
        presence masks preserve that distinction.

        Unsupported markers identify an encoding limitation, not preserved
        original scientific values. Column metadata is stored in:
            trials.attrs["columnMetadata"]

        Current feature limits
        ----------------------
        includeTriggers : bool, default True
            Wide trigger-column expansion is not implemented. If trigger
            records are detected anywhere in the comprehensive matrix,
            this request warns and returns None.

            Use includeTriggers=False to generate a table without wide
            trigger columns. Trigger timing references STILL WORK with
            includeTriggers=False; source trigger records are not removed.
        includeConfiguration : bool, default False
            Configuration projection is not implemented.
            True produces a warning and returns None.

        Construction rejects empty datasets and nonbehavioral members rather
        than silently skipping them. Incompatible cross-member encodings
        also raise rather than silently coercing scientific values.

        Storage options
        ---------------
        settings, settingsName:
            Select PGL configuration. Supply at most one.
            Without an override, use the dataset's bound runtime settings.
            A conflicting database identity is rejected.
        storageOptionsByBackend:
            Runtime payload-access options keyed by backend ID.
            Uses the dataset's loaded runtime options when omitted.
        sslRootCert:
            Optional database TLS certificate configuration.
        cacheDirectory:
            Cache directory override when member-session loading requires it.
        directLocalPolicy:
            Member loading policy: "materialize" or "immutable".
            None retains the dataset's configured load policy or the loader
            default. "immutable" requires source paths/files to remain unchanged.

        cacheDirectory and directLocalPolicy apply to missing-product
        construction, not to opening an existing behavior matrix product.

        Examples
        --------
        trials = dataset.getBehaviorTable(taskName="memoryDescriptionTask")

        # All tasks, without unimplemented wide trigger expansion.
        trials = dataset.getBehaviorTable(taskSelection="all", includeTriggers=False)

        # Relative to recorded segment 1 START, not verified visual onset.
        trials = dataset.getBehaviorTable(taskName="memoryDescriptionTask", responseReference="segment", segmentNum=1)

        # Trigger timing without wide trigger columns.
        trials = dataset.getBehaviorTable(taskName="memoryDescriptionTask", responseReference="trigger", triggerName="stimulusOnset", includeTriggers=False)

        trials = dataset.getBehaviorTable(taskName="memoryDescriptionTask", includeAllColumns=True)
        trials = dataset.getBehaviorTable(taskName="memoryDescriptionTask", columns=["sourceCheckpointID", "subjectID", "sessionName", "runName", "sourceTrialNum", "responseTime"])

        if trials is not None:
            print(trials.head())

        Side effects, freshness, and persistence
        ----------------------------------------
        First access may read saved member sources and attach a working matrix,
        even if subsequent table-option validation rejects the request.
        Opening a saved product reference replaces it with a cached working
        matrix in dataset.data.

        Existing products are construction snapshots. Adding/removing members,
        changing aliases or labels, or changing construction code does not
        automatically rebuild an attached behavior product.

        Construction currently retains extracted member matrices in memory
        until assembly completes; it is not bounded-memory streaming.
        Projection may materialize arrays, and pandas results are in memory.

        Editing the returned DataFrame does not modify the source matrix.
        Does not automatically save database records or Zarr payloads.
        Use dataset.save() to persist attached products explicitly.
        """
        from .pglBehavior import pglBehavior

        matrix = self.getBehaviorData(settings=settings, settingsName=settingsName, storageOptionsByBackend=storageOptionsByBackend, sslRootCert=sslRootCert, cacheDirectory=cacheDirectory, directLocalPolicy=directLocalPolicy)

        return pglBehavior.getBehaviorTable(matrix, taskName=taskName,**options)