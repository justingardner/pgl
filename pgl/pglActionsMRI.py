################################################################
#   filename: pglActionsMRI.py
#    purpose: actions for processing MRI data
#         by: JLG
#       date: Nov 15, 2026
################################################################

#############
# Imports
#############
from copy import deepcopy

from .pglMessages import pglMessages
from .pglPipeline import pglActionCollection

from traitlets import Bool, Float, Instance, Int, List, Unicode

from .pglPipeline import pglAction, pglDeferredAction, pglProductSelector


################################################################
# MRI actions
################################################################
class pglActionsMRI(pglActionCollection):
    """MRI preprocessing actions.

    Private preparation helpers do not execute tools, change image geometry,
    publish products, or append action history.
    """

    ################################################################
    # Prepare geometry groups and explicit run-alignment references
    ################################################################
    @staticmethod
    def _prepareRunAlignment(mri, scanIndices=None, *, referenceRun="middle", allowGeometryMismatch=False, spatialToleranceMM=0.001, orientationTolerance=0.000001):
        """Prepare the inputs for the run-alignment action.

        scanIndices:
            Zero-based indices in session.mri.scans.
            None selects all attached scans; no implicit functional-scan
            classification or acquisition-time sorting is performed.

        referenceRun:
            "first", "middle", or "last", independently within each group.
            An integer is an explicit index in session.mri.scans.
            A list provides one choice per geometry group.
            Middle means len(group) // 2, the later middle for even counts.

        Geometry:
            Uses preserved acquisition-header metadata, or reads the source
            header when no original snapshot exists.
            Never reads editable metadata or accesses scan.data.

            Requires a coded qform and recognized spatial units.
            A coded sform must agree with qform numerically.
            Code meanings are retained; a coded form alone is not proof that
            its coordinates faithfully represent the physical acquisition.

            Compares spatial shape, voxel sizes, orientation, origin, and
            voxel-center corner positions. Unit conversion is for comparison
            only and never changes the image/header.

            Tolerances are absolute. spatialToleranceMM is in millimeters;
            orientationTolerance applies to normalized axis components.

        allowGeometryMismatch:
            False forms independent geometry-compatible alignment groups.
            True requires one explicit selected reference index and authorizes
            one cross-geometry alignment group. Application must then resolve
            its output grid explicitly if the selected geometries differ.

        Returns:
            A JSON-compatible resolved plan for the action to record.

        Errors raise internally to the action runner. Multiple independent
        groups, or an authorized mismatch, warn at level=1 and continue.

        Does not estimate transforms, select reference volumes, read voxel
        arrays, or mutate scans. Backing filesystems may independently verify
        or spool source files when their header streams are opened.

        Concurrent modification of scans, sources, or metadata is unsupported.
        """
        import math
        import numpy as np

        from .pglMRI import pglMRI, pglMRINiftiMetadata

        if not isinstance(mri, pglMRI): raise TypeError("Expected pglMRI.")
        if type(allowGeometryMismatch) is not bool:
            raise TypeError("allowGeometryMismatch must be Boolean.")

        for name, value in (
            ("spatialToleranceMM", spatialToleranceMM),
            ("orientationTolerance", orientationTolerance),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")

        scans = list(mri.scans)
        if not scans: raise ValueError("No MRI scans are attached.")

        if scanIndices is None:
            indices = list(range(len(scans)))
        elif isinstance(scanIndices, (list, tuple)):
            indices = list(scanIndices)
        else:
            raise TypeError("scanIndices must be a list/tuple of indices or None.")

        if not indices: raise ValueError("Select at least one MRI scan.")
        if any(type(index) is not int or not 0 <= index < len(scans) for index in indices):
            raise ValueError("scanIndices must contain valid zero-based session.mri.scans indices.")
        if len(set(indices)) != len(indices):
            raise ValueError("scanIndices must not contain duplicates.")

        identifiers = [scans[index].uuid for index in indices]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Select each acquisition UUID only once.")

        def validChoice(value):
            return type(value) is int or (
                isinstance(value, str) and value in ("first", "middle", "last")
            )

        if isinstance(referenceRun, list):
            if not referenceRun or not all(validChoice(value) for value in referenceRun):
                raise ValueError("Invalid per-group referenceRun choices.")
        elif not validChoice(referenceRun):
            raise ValueError("referenceRun must be an index, first/middle/last, or a list of those choices.")

        if allowGeometryMismatch and type(referenceRun) is not int:
            raise ValueError("Cross-geometry alignment requires an explicit referenceRun index.")

        geometries = {}
        scanRecords = []

        for index in indices:
            scan = scans[index]
            snapshot = scan.sourceMetadata.get("nifti")

            # Acquisition geometry remains authoritative even when a working
            # image or editable metadata contains different header values.
            if snapshot is None:
                if scan._backing is not None:
                    snapshot = pglMRINiftiMetadata.readBacking(scan._backing)
                elif scan.localPath:
                    snapshot = pglMRINiftiMetadata.read(scan.localPath)
                else:
                    raise ValueError(f"Scan {index} has no acquisition header or source.")

            if not isinstance(snapshot, dict):
                raise ValueError(f"Scan {index}: invalid acquisition-header snapshot.")

            fields = snapshot.get("fields")
            transforms = snapshot.get("transforms")
            if not isinstance(fields, dict) or not isinstance(transforms, dict):
                raise ValueError(f"Scan {index}: incomplete acquisition-header snapshot.")

            dimensions = fields.get("dim")
            if (
                not isinstance(dimensions, list) or not dimensions
                or type(dimensions[0]) is not int or dimensions[0] not in (3, 4)
                or len(dimensions) <= dimensions[0]
                or any(type(size) is not int or size <= 0 for size in dimensions[1:dimensions[0] + 1])
            ):
                raise ValueError(f"Scan {index}: select a valid 3D/4D image; higher axes are not implicitly collapsed.")

            shape = dimensions[1:4]
            volumeCount = dimensions[4] if dimensions[0] == 4 else 1
            units = fields.get("xyzt_units")
            factor = {1: 1000.0, 2: 1.0, 3: 0.001}.get(units & 7) if type(units) is int else None
            if factor is None:
                raise ValueError(f"Scan {index}: unknown or unsupported spatial units.")

            qformCode = fields.get("qform_code")
            sformCode = fields.get("sform_code")
            if type(qformCode) is not int or qformCode <= 0:
                raise ValueError(f"Scan {index}: a coded acquisition qform is required; geometry will not be guessed.")
            if type(sformCode) is not int or sformCode < 0:
                raise ValueError(f"Scan {index}: invalid sform code.")

            for form, code in (("qform", qformCode), ("sform", sformCode)):
                if transforms.get(f"{form}_code", code) != code:
                    raise ValueError(f"Scan {index}: inconsistent recorded {form} code.")

            def readAffine(value, label):
                """Interpret a recorded affine in mm without editing its source."""
                try:
                    matrix = np.asarray(value, dtype=float)
                except (TypeError, ValueError):
                    raise ValueError(f"Scan {index}: invalid {label}.") from None

                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError(f"Scan {index}: invalid or nonfinite {label}.")
                if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
                    raise ValueError(f"Scan {index}: invalid {label} homogeneous row.")

                matrix = matrix.copy()
                matrix[:3] *= factor
                if np.linalg.matrix_rank(matrix[:3, :3]) != 3:
                    raise ValueError(f"Scan {index}: singular {label}.")
                return matrix

            qform = readAffine(transforms.get("qform"), "qform")
            sizes = np.linalg.norm(qform[:3, :3], axis=0)
            directions = qform[:3, :3] / sizes
            origin = qform[:3, 3]
            corners = np.array([
                [x, y, z, 1.0]
                for x in (0, shape[0] - 1)
                for y in (0, shape[1] - 1)
                for z in (0, shape[2] - 1)
            ]).T

            pixelDimensions = fields.get("pixdim")
            if not isinstance(pixelDimensions, list) or len(pixelDimensions) < 4:
                raise ValueError(f"Scan {index}: missing voxel sizes.")

            spacing = np.asarray(pixelDimensions[1:4], dtype=float) * factor
            if (
                not np.isfinite(spacing).all() or np.any(spacing <= 0)
                or not np.allclose(spacing, sizes, rtol=0, atol=spatialToleranceMM)
            ):
                raise ValueError(f"Scan {index}: pixdim and qform voxel sizes disagree.")

            if sformCode > 0:
                sform = readAffine(transforms.get("sform"), "sform")
                sSizes = np.linalg.norm(sform[:3, :3], axis=0)
                sDirections = sform[:3, :3] / sSizes
                cornerError = np.linalg.norm(((sform - qform) @ corners)[:3], axis=0).max()

                if (
                    not np.allclose(sSizes, sizes, rtol=0, atol=spatialToleranceMM)
                    or not np.allclose(sDirections, directions, rtol=0, atol=orientationTolerance)
                    or cornerError > spatialToleranceMM
                ):
                    raise ValueError(
                        f"Scan {index}: coded sform conflicts with acquisition qform. "
                        "Resolve the geometry before registration; allowGeometryMismatch "
                        "does not bypass conflicting headers within an image."
                    )

            geometries[index] = {
                "shape": shape,
                "matrix": qform,
                "sizes": sizes,
                "directions": directions,
                "origin": origin,
                "corners": corners,
            }
            scanRecords.append({
                "scanIndex": index,
                "scanUUID": scan.uuid,
                "scanName": scan.name,
                "spatialShape": list(shape),
                "volumeCount": volumeCount,
                "qformCode": qformCode,
                "sformCode": sformCode,
                "qformMM": qform.tolist(),
                "voxelSizesMM": sizes.tolist(),
            })

        def sameGrid(leftIndex, rightIndex):
            left, right = geometries[leftIndex], geometries[rightIndex]
            if left["shape"] != right["shape"]: return False
            if not np.allclose(left["sizes"], right["sizes"], rtol=0, atol=spatialToleranceMM): return False
            if not np.allclose(left["origin"], right["origin"], rtol=0, atol=spatialToleranceMM): return False
            if not np.allclose(left["directions"], right["directions"], rtol=0, atol=orientationTolerance): return False

            error = np.linalg.norm(
                ((left["matrix"] - right["matrix"]) @ left["corners"])[:3],
                axis=0,
            ).max()
            return bool(error <= spatialToleranceMM)

        # Comparing every member prevents a chain of small differences from
        # placing mutually incompatible endpoints in the same group.
        groups = []
        for index in indices:
            matches = [
                position for position, members in enumerate(groups)
                if all(sameGrid(index, other) for other in members)
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"Scan {index} matches multiple groups near the tolerance boundary. "
                    "Review the selection/tolerance rather than assigning it arbitrarily."
                )
            if matches: groups[matches[0]].append(index)
            else: groups.append([index])

        def resolveReference(choice, members):
            if type(choice) is int:
                if choice not in members:
                    raise ValueError(
                        f"Reference scan {choice} is not in group {members}. "
                        "Specify one reference per group or explicitly authorize "
                        "cross-geometry alignment with a selected reference index."
                    )
                return choice
            position = {"first": 0, "middle": len(members) // 2, "last": len(members) - 1}[choice]
            return members[position]

        if allowGeometryMismatch:
            reference = resolveReference(referenceRun, indices)
            alignmentGroups = [{
                "groupIndex": 0,
                "scanIndices": list(indices),
                "referenceRule": referenceRun,
                "referenceScanIndex": reference,
                "referenceScanUUID": scans[reference].uuid,
            }]
        else:
            rules = referenceRun if isinstance(referenceRun, list) else [referenceRun] * len(groups)
            if len(rules) != len(groups):
                raise ValueError(f"Expected {len(groups)} reference choices, received {len(rules)}.")

            alignmentGroups = []
            for groupIndex, (members, rule) in enumerate(zip(groups, rules)):
                reference = resolveReference(rule, members)
                alignmentGroups.append({
                    "groupIndex": groupIndex,
                    "scanIndices": list(members),
                    "referenceRule": rule,
                    "referenceScanIndex": reference,
                    "referenceScanUUID": scans[reference].uuid,
                })

        crossGeometry = allowGeometryMismatch and len(groups) > 1
        plan = {
            "schemaVersion": 1,
            "kind": "mriRunAlignmentPreparation",
            "scanIndices": list(indices),
            "requestedReference": deepcopy(referenceRun),
            "allowGeometryMismatch": allowGeometryMismatch,
            "geometryPolicy": {
                "source": "originalQform",
                "spatialToleranceMM": float(spatialToleranceMM),
                "orientationTolerance": float(orientationTolerance),
                "compareSpatialShape": True,
                "compareFieldOfViewCorners": True,
                "groupOrder": "firstAppearanceInSelectedOrder",
                "middleRule": "len(group)//2",
            },
            "scans": scanRecords,
            "geometryGroups": [
                {"groupIndex": index, "scanIndices": list(members)}
                for index, members in enumerate(groups)
            ],
            "alignmentGroups": alignmentGroups,
            "requiresExplicitOutputGrid": crossGeometry,
            "referenceImageResolution": "Required from explicit source/motion product before estimation.",
        }

        if len(groups) > 1:
            message = (
                "Cross-geometry alignment was explicitly authorized. "
                "Application requires an explicit output-grid choice."
                if crossGeometry else
                f"Selected scans contain {len(groups)} acquisition-grid groups. "
                "Alignment will remain independent within each group."
            )
            pglMessages.warning(message, level=1)

        return plan
    
    ################################################################
    # Validate a recorded within-run motion product
    ################################################################
    @staticmethod
    def _validateMotionProduct(record):
        """Validate native or interpreted motion metadata without reading files.

        Returns independent scan records indexed by acquisition UUID.
        Uses the same contract as product inspection and motion plotting.

        Invalid definitions propagate to the owning action runner.
        """
        from .pglMRI import pglMRITransforms

        return pglMRITransforms.validateMotionProduct(record)
    
    ################################################################
    # Resolve explicit motion dependencies and prepare run alignment
    ################################################################
    @staticmethod
    def _prepareRunAlignmentInputs(session, withinRunMotion, scanIndices=None, *, referenceRun="middle", allowGeometryMismatch=False, spatialToleranceMM=0.001, orientationTolerance=0.000001):
        """Prepare run alignment from an explicitly selected motion product.

        Called internally by the run-alignment action. Errors propagate to
        its ordinary action runner, which warns and stops execution.

        withinRunMotion:
            A pglProductSelector for an existing rigidMotion product.
            No latest-product search, implicit producer execution, or fallback
            to another source stage is performed.

        scanIndices:
            Current zero-based indices in session.mri.scans.
            None selects all attached scans, matching the geometry helper.
            The motion product must cover every selected scan.

        Matches product scan records by UUID, not historical scan position.
        Historical indices remain provenance and are returned separately.

        Returns:
            A portable plan identifying geometry groups, reference scans,
            per-scan reference volumes, source requirements, and required
            transform artifacts.

        Source requirements are NOT resolved payloads. A sourceStageID and
        content hash alone do not establish how to open the source image.
        The scientific action must explicitly resolve those stages and verify
        their bytes and geometry before estimating alignment.

        Product resolution automatically records exact input consumption when
        called inside an action. This helper does not append a separate action
        history entry, publish products, execute AFNI, or apply transforms.

        Native AFNI matrix mapping remains unverified. This helper does not
        authorize composition/application of those matrices.
        """
        from .pglPipeline import pglProductSelector
        from .pglSession import pglSession

        if not isinstance(session, pglSession) or session.mri is None:
            raise ValueError("Run alignment requires a session with MRI scans.")
        if not isinstance(withinRunMotion, pglProductSelector):
            raise TypeError("withinRunMotion must be an explicit pglProductSelector.")

        # Resolve through the selector so execution history captures the exact
        # dependency. No file is read by product resolution itself.
        product = withinRunMotion.resolve(session)
        if product is None:
            raise ValueError("The required within-run motion product could not be resolved.")

        productRecord = product.record
        motionByScan = pglActionsMRI._validateMotionProduct(productRecord)

        plan = pglActionsMRI._prepareRunAlignment(
            session.mri,
            scanIndices,
            referenceRun=referenceRun,
            allowGeometryMismatch=allowGeometryMismatch,
            spatialToleranceMM=spatialToleranceMM,
            orientationTolerance=orientationTolerance,
        )

        missing = [
            record["scanIndex"]
            for record in plan["scans"]
            if record["scanUUID"] not in motionByScan
        ]
        if missing:
            raise ValueError(
                f"Within-run motion product {product.name!r} does not cover "
                f"selected MRI scan indices {missing}."
            )

        requirements = []
        requirementsByIndex = {}

        for geometry in plan["scans"]:
            scanIndex = geometry["scanIndex"]
            scanUUID = geometry["scanUUID"]
            motion = motionByScan[scanUUID]
            volumeIndices = motion["volumeIndices"]

            # These are original acquisition indices, not row numbers in a
            # shortened temporal stage or motion-matrix file.
            if any(index >= geometry["volumeCount"] for index in volumeIndices):
                raise ValueError(
                    f"Motion product {product.name!r} refers to an original "
                    f"volume outside scan {scanIndex}'s acquisition."
                )

            referenceVolumeIndex = motion["referenceVolumeIndex"]
            matrixRole = motion["matricesRole"]
            parameterRole = motion["motionParametersRole"]

            requirement = {
                "scanIndex": scanIndex,
                "scanUUID": scanUUID,
                "recordedProducerScanIndex": motion["scanIndex"],
                "sourceStageID": motion["sourceStageID"],
                "sourceIdentity": deepcopy(motion["sourceIdentity"]),
                "volumeIndices": list(volumeIndices),
                "referenceVolumeIndex": referenceVolumeIndex,
                "referenceRow": volumeIndices.index(referenceVolumeIndex),
                "matrixArtifact": deepcopy(productRecord["artifacts"][matrixRole]),
                "motionParametersArtifact": deepcopy(productRecord["artifacts"][parameterRole]),
            }

            requirements.append(requirement)
            requirementsByIndex[scanIndex] = requirement

        alignmentGroups = []

        for group in plan["alignmentGroups"]:
            reference = requirementsByIndex[group["referenceScanIndex"]]

            alignmentGroups.append({
                **deepcopy(group),
                "referenceSourceStageID": reference["sourceStageID"],
                "referenceSourceIdentity": deepcopy(reference["sourceIdentity"]),
                "referenceVolumeIndex": reference["referenceVolumeIndex"],
                "referenceRow": reference["referenceRow"],
            })

        plan["kind"] = "mriRunAlignmentInputs"
        plan["alignmentGroups"] = alignmentGroups
        plan["withinRunMotion"] = {
            "selector": withinRunMotion.toJSONdict(),
            "productID": product.productID,
            "producerActionID": product.producerActionID,
            "matrixConvention": deepcopy(productRecord["metadata"]["matrixConvention"]),
        }
        plan["sourceRequirements"] = requirements
        plan["referenceImageResolution"] = "Explicit source-stage payload resolution required before estimation."
        plan["readyForEstimation"] = False

        return plan
    
    ################################################################
    # Resolve an explicitly declared acquisition or derived image
    ################################################################
    @staticmethod
    def _resolveImageSource(session, source, *, expectedMotionSource=None, cacheDirectory=None):
        """Resolve a verified local image and its original-volume mapping.

        Internal action helper: failures raise to the action runner.
        No fallback source selection, regeneration, or header editing occurs.

        source:
            Acquisition:
                schemaVersion, kind="acquisition", scanUUID

            Derived:
                schemaVersion, kind="derived", scanUUID,
                productID, artifactRole

        Derived image products must declare:
            metadata["imageStages"][artifactRole] = {
                "scanUUID": ...,
                "volumeIndices": [...],
            }

        volumeIndices describes EVERY volume stored in that image, in file
        order. It is not a request to remove or reorder volumes.

        Stage identities:
            acquisition:<scanUUID>
            product:<productID>:<artifactRole>

        expectedMotionSource:
            Optional scan record from a validated motion product. Its scan UUID,
            stage ID, content identity, and original-volume mapping must match.

        cacheDirectory:
            Local cache override for deferred acquisition materialization or
            derived-file retrieval. None uses the source's existing cache
            configuration.

        Returns:
            Runtime dictionary containing localPath, source identity, original
            volume mapping, and the actual on-disk header snapshot.

        Reads file bytes for hashing and headers, not voxel arrays.
        External tools subsequently read voxel data when explicitly invoked.
        Sources must remain unchanged while readers/actions depend on them.

        This establishes source identity and declared volume correspondence.
        Geometry compatibility with acquisition/reference grids must still be
        checked by the scientific action before estimation/application.

        Does not replace scan.localPath, scan._backing, scan._data, or metadata.
        """
        from pathlib import Path

        from .pglExternalTools import _copyProvenance, pglExternalTool
        from .pglMRI import pglMRINiftiMetadata
        from .pglPipeline import pglProductSelector
        from .pglSession import pglSession

        if not isinstance(session, pglSession) or session.mri is None:
            raise ValueError("Image-source resolution requires a session with MRI scans.")

        source = _copyProvenance(source)
        if type(source) is not dict:
            raise TypeError("source must be an explicit image-source dictionary.")
        if type(source.get("schemaVersion")) is not int or source["schemaVersion"] != 1:
            raise ValueError("Unsupported image-source version.")

        kind = source.get("kind")
        expectedFields = {"schemaVersion", "kind", "scanUUID"}
        if kind == "derived":
            expectedFields |= {"productID", "artifactRole"}
        elif kind != "acquisition":
            raise ValueError("Image-source kind must be acquisition or derived.")

        if set(source) != expectedFields:
            raise ValueError("Invalid image-source fields.")

        scanUUID = source["scanUUID"]
        if not isinstance(scanUUID, str) or not scanUUID.strip():
            raise ValueError("Image source requires a scan UUID.")

        matches = [
            (index, scan)
            for index, scan in enumerate(session.mri.scans)
            if scan.uuid == scanUUID
        ]
        if len(matches) != 1:
            raise ValueError("Image source scan UUID is absent or ambiguous in session.mri.scans.")

        scanIndex, scan = matches[0]
        expectedIdentity = None
        volumeIndices = None
        productID = None

        if kind == "acquisition":
            stageID = f"acquisition:{scanUUID}"

            # Access the acquisition backing directly. scan.data could expose
            # an edited working image and is not the source contract here.
            if scan._backing is not None:
                cache = scan._cacheDirectory if cacheDirectory is None else cacheDirectory
                if cache is None or not str(cache).strip():
                    raise ValueError("A local cache directory is required for this acquisition source.")

                path = Path(scan._backing.materialize(cache))
                expectedIdentity = scan._backing.expectedIdentity

            elif scan.localPath:
                path = Path(scan.localPath).expanduser().absolute()
            else:
                raise ValueError(f"Scan {scanIndex} has no acquisition source.")

        else:
            productID = source["productID"]
            role = source["artifactRole"]

            for label, value in (("productID", productID), ("artifactRole", role)):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{label} must be a nonblank string.")

            # Pinned resolution automatically records exact product consumption
            # when invoked inside an action.
            selector = pglProductSelector(productID=productID)
            product = selector.resolve(session)
            if product is None:
                raise ValueError("The explicitly selected image product is unavailable.")

            artifact = product.artifacts.get(role)
            if not isinstance(artifact, dict):
                raise ValueError(f"Image product has no artifact role {role!r}.")

            imageStages = product.metadata.get("imageStages")
            stage = imageStages.get(role) if isinstance(imageStages, dict) else None

            if not isinstance(stage, dict) or set(stage) != {"scanUUID", "volumeIndices"}:
                raise ValueError("Derived image product lacks an explicit image-stage association.")
            if stage["scanUUID"] != scanUUID:
                raise ValueError("Derived image belongs to a different acquisition scan.")

            volumeIndices = stage["volumeIndices"]
            if (
                type(volumeIndices) is not list or not volumeIndices
                or any(type(index) is not int or index < 0 for index in volumeIndices)
                or len(set(volumeIndices)) != len(volumeIndices)
            ):
                raise ValueError("Derived image volumeIndices must contain unique nonnegative original indices.")

            attachmentName = artifact["attachmentName"]
            attachment = session.derivedFiles.get(attachmentName)
            if attachment is None:
                raise ValueError("The declared image payload is not attached; it may have been discarded.")

            expectedIdentity = {key: artifact[key] for key in ("sha256", "size_bytes")}
            if any(attachment.get(key) != value for key, value in expectedIdentity.items()):
                raise ValueError("The image attachment no longer matches its recorded product identity.")

            path = session.getDerivedFile(attachmentName, cacheDirectory=cacheDirectory)
            if path is None:
                raise ValueError("The declared image payload could not be retrieved.")

            path = Path(path)
            stageID = f"product:{productID}:{role}"

        if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
            raise ValueError("Resolved image source must be an existing .nii or .nii.gz file.")

        # Verify the actual file, including local-only acquisition sources and
        # callers whose retrieval implementation returned an unverified path.
        identity = pglExternalTool._fileIdentity(path)
        if expectedIdentity is not None and identity != expectedIdentity:
            raise ValueError("Resolved image bytes differ from their expected identity.")

        snapshot = pglMRINiftiMetadata.read(path)
        dimensions = snapshot["fields"].get("dim")
        if (
            not isinstance(dimensions, list) or not dimensions
            or type(dimensions[0]) is not int or dimensions[0] not in (3, 4)
            or len(dimensions) <= dimensions[0]
            or any(type(size) is not int or size <= 0 for size in dimensions[1:dimensions[0] + 1])
        ):
            raise ValueError("Image-source resolution supports valid 3D/4D images only.")

        volumeCount = dimensions[4] if dimensions[0] == 4 else 1

        if kind == "acquisition":
            original = scan.sourceMetadata.get("nifti")
            if original is not None and original != snapshot:
                raise ValueError("Acquisition header differs from its preserved original snapshot.")
            volumeIndices = list(range(volumeCount))

        elif len(volumeIndices) != volumeCount:
            raise ValueError(
                f"Derived image stores {volumeCount} volumes but declares "
                f"{len(volumeIndices)} original-volume indices."
            )

        if expectedMotionSource is not None:
            expected = _copyProvenance(expectedMotionSource)
            if type(expected) is not dict:
                raise TypeError("expectedMotionSource must be a motion scan record.")

            comparisons = {
                "scanUUID": scanUUID,
                "sourceStageID": stageID,
                "sourceIdentity": identity,
                "volumeIndices": volumeIndices,
            }
            for field, actual in comparisons.items():
                if expected.get(field) != actual:
                    raise ValueError(f"Resolved image does not match the motion product's {field}.")

        return {
            "schemaVersion": 1,
            "source": source,
            "scanIndex": scanIndex,
            "scanUUID": scanUUID,
            "sourceStageID": stageID,
            "sourceIdentity": identity,
            "volumeIndices": list(volumeIndices),
            "volumeCount": volumeCount,
            "localPath": path,
            "nifti": snapshot,
            "productID": productID,
        }
        
    ################################################################
    # Apply interpreted within-run motion on original acquisition grids
    ################################################################
    @staticmethod
    def _applyWithinRunMotion(action, session, productRecord, *, outputName, cacheDirectory=None):
        """Apply a version-2 motion product and register corrected images.

        Initial scope:
            Acquisition sources only.
            Full original volume sequence.
            Within-run shared-grid rigid motion.
            Cubic SciPy interpolation, constant-zero boundary, float32 output.

        Validates all declared sources/artifacts before correction.
        Writes new images with original acquisition geometry.
        Does not modify original scans or replace their backing/data objects.

        Native matrices and interpreted matrices must agree. Output geometry
        and source identities are verified before attachment.

        Returns the registered image-product record.
        Internal errors propagate to the action runner.
        """
        import tempfile
        from pathlib import Path

        import nibabel as nib
        import numpy as np
        from scipy.ndimage import affine_transform

        from .pglExternalTools import pglExternalTool
        from .pglMRI import pglMRITransforms

        records = pglMRITransforms.validateMotionProduct(productRecord)
        if productRecord["metadata"]["schemaVersion"] != 2:
            raise ValueError("Application requires interpreted version-2 motion metadata.")

        # Reject a duplicate output before generating any files.
        for entry in session.actionHistory:
            if any(
                record.get("name") == outputName and record.get("recordStatus") == "committed"
                for record in entry.actionProducts
            ):
                raise ValueError(f"Product {outputName!r} already exists.")

        jobs = []

        for scanUUID, record in records.items():
            if record["sourceStageID"] != f"acquisition:{scanUUID}":
                raise ValueError(
                    "This initial application engine supports acquisition sources only. "
                    "Derived source stages require explicit support."
                )

            source = pglActionsMRI._resolveImageSource(
                session,
                {"schemaVersion": 1, "kind": "acquisition", "scanUUID": scanUUID},
                expectedMotionSource=record,
                cacheDirectory=cacheDirectory,
            )

            # Confirm registered native artifacts still match the interpreted
            # product before using its derived sampling matrices.
            for field, columns in (("matricesRole", 12), ("motionParametersRole", 6)):
                artifact = productRecord["artifacts"][record[field]]
                attachment = session.derivedFiles.get(artifact["attachmentName"])
                expected = {key: artifact[key] for key in ("sha256", "size_bytes")}

                if attachment is None or any(attachment.get(key) != value for key, value in expected.items()):
                    raise ValueError("Motion attachment is missing or differs from the recorded product.")

                path = session.getDerivedFile(artifact["attachmentName"], cacheDirectory=cacheDirectory)
                if path is None or pglExternalTool._fileIdentity(path) != expected:
                    raise ValueError("Motion artifact failed verification.")

                values = np.loadtxt(path, comments="#", ndmin=2)
                if values.shape != (source["volumeCount"], columns) or not np.isfinite(values).all():
                    raise ValueError("Invalid motion artifact dimensions or values.")

                if field == "matricesRole":
                    nativeRows = np.array([
                        np.asarray(transform["matrix"])[:3].reshape(-1)
                        for transform in record["interpretation"]["nativeTransforms"]
                    ])
                    if not np.allclose(values, nativeRows, rtol=0, atol=1e-10):
                        raise ValueError("Native matrix bytes disagree with interpreted transforms.")

            image = nib.load(source["localPath"])
            physical = image.get_qform(coded=True)[0]
            interpretedFrame = record["interpretation"]["voxelFrame"]

            if physical is None or image.header.get_xyzt_units()[0] != "mm":
                raise ValueError("Application requires a coded millimeter acquisition qform.")
            if (
                list(image.shape[:3]) != interpretedFrame["spatialShape"]
                or not np.allclose(physical, interpretedFrame["voxelToPhysicalMM"], rtol=0, atol=0.0001)
            ):
                raise ValueError("Motion interpretation does not describe this source grid.")

            jobs.append((record, source))

        parent = None
        if cacheDirectory is not None:
            parent = Path(cacheDirectory).expanduser().resolve()
            parent.mkdir(parents=True, exist_ok=True)

        outputDirectory = Path(tempfile.mkdtemp(
            prefix="pgl-motion-applied-",
            dir=None if parent is None else str(parent),
        ))
        completed = []

        # Retain failed staging for diagnosis. No partial image product is
        # registered; the action runner marks any failed session unsafe.
        for record, source in jobs:
            image = nib.load(source["localPath"])
            shape = image.shape
            corrected = np.empty(shape, dtype=np.float32)

            # Process input volumes individually; output is currently held in
            # memory. Out-of-core output writing can be added separately.
            for row, transform in enumerate(record["interpretation"]["voxelTransforms"]):
                mapping = np.asarray(transform["matrix"])
                volume = np.asarray(image.dataobj[..., row], dtype=np.float32)
                if not np.isfinite(volume).all():
                    raise ValueError("Source image contains nonfinite voxel values.")

                corrected[..., row] = affine_transform(
                    volume,
                    matrix=mapping[:3, :3],
                    offset=mapping[:3, 3],
                    output_shape=shape[:3],
                    order=3,
                    mode="constant",
                    cval=0,
                    prefilter=True,
                )

            if not np.isfinite(corrected).all():
                raise ValueError("Correction produced nonfinite values.")

            header = image.header.copy()
            header.set_data_dtype(np.float32)
            header.set_slope_inter(1, 0)
            output = image.__class__(corrected, image.affine.copy(), header=header)

            # Geometry is specified before writing, on the grid used during
            # sampling—not copied onto an output from a different lattice.
            output.set_qform(image.get_qform(), code=int(image.header["qform_code"]))
            output.set_sform(image.get_sform(), code=int(image.header["sform_code"]))

            filename = outputDirectory / f"scan-{record['scanUUID']}.nii.gz"
            nib.save(output, filename)
            stored = nib.load(filename)

            if (
                stored.shape != image.shape
                or stored.header.get_xyzt_units() != image.header.get_xyzt_units()
                or not np.allclose(stored.header.get_zooms(), image.header.get_zooms(), rtol=0, atol=0.0001)
            ):
                raise ValueError("Corrected image dimensions, spacing, TR, or units changed.")

            for form in ("qform", "sform"):
                before, beforeCode = getattr(image, f"get_{form}")(coded=True)
                after, afterCode = getattr(stored, f"get_{form}")(coded=True)
                if int(beforeCode) != int(afterCode):
                    raise ValueError(f"Corrected image {form} code changed.")
                if before is not None and (after is None or not np.allclose(before, after, rtol=0, atol=0.0001)):
                    raise ValueError(f"Corrected image {form} geometry changed.")

            if pglExternalTool._fileIdentity(source["localPath"]) != source["sourceIdentity"]:
                raise ValueError("Acquisition source changed during correction.")

            completed.append((record, filename))

        files = {}
        imageStages = {}
        operationID = action.actionHistory[-1].actionID

        for record, filename in completed:
            role = f"scan.{record['scanUUID']}.image"
            attachmentName = f"mri.{operationID}.{role}"
            session.addDerivedFile(attachmentName, filename, mediaType="application/octet-stream")
            files[role] = attachmentName
            imageStages[role] = {
                "scanUUID": record["scanUUID"],
                "volumeIndices": list(record["volumeIndices"]),
            }

        return action.registerProduct(
            outputName,
            "image",
            session=session,
            files=files,
            inputProductIDs=[productRecord["productID"]],
            metadata={
                "schemaVersion": 1,
                "operation": "withinRunMotionApplication",
                "imageStages": imageStages,
                "appliedMotionProductID": productRecord["productID"],
                "appliedProducerActionID": productRecord["producerActionID"],
                "appliedTransformIDs": [
                    transform["transformID"]
                    for record in records.values()
                    for transform in record["interpretation"]["voxelTransforms"]
                ],
                "outputGrid": "originalAcquisition",
                "sampling": {
                    "implementation": "scipy.ndimage.affine_transform",
                    "order": 3,
                    "mode": "constant",
                    "cval": 0.0,
                    "prefilter": True,
                    "outputDtype": "float32",
                },
            },
            invocationIDs=[],
        )

    ################################################################
    # Present a completed motion product
    ################################################################
    @staticmethod
    def _showMotionProduct(session, productID):
        """Display completed motion results without repeating estimation."""
        import matplotlib.pyplot as plt

        product = session.getProduct(productID=productID)
        if product is None: return
        figures = product.display()
        if figures:
            plt.show(block=False)
            
    ################################################################
    # Estimate within-run motion using AFNI
    ################################################################
    class mriEstimateMotionAFNI(pglDeferredAction):
        """Estimate rigid motion and optionally correct original acquisitions.

        scanIndices are zero-based session.mri.scans indices.
        None selects all attached scans; nonfunctional scans are not silently
        skipped or classified from their names.

        This initial action accepts original acquisition sources only.
        No dummy removal or temporal correction is implicit.

        Default execution estimates and applies within-run correction.
        deferApply=True registers motion without creating corrected images.

        Corrected data remain on each original acquisition grid.
        """
        scanIndices = List(Int(), default_value=None, allow_none=True, help="Zero-based MRI scan indices; None selects all.")
        referenceVolume = Int(0, min=0, help="Reference volume index within each original acquisition.")
        twoPass = Bool(True, help="Use AFNI two-pass motion estimation.")
        outputName = Unicode("motion", help="Name of the motion-estimation product.")
        correctedOutputName = Unicode("motionCorrected", help="Name of the corrected-image product.")
        showMotion = Bool(True, help="Display motion plots after successful execution.")
        afniDirectory = Unicode(default_value=None, allow_none=True, help="AFNI executable directory; None uses ordinary discovery.")
        cacheDirectory = Unicode(default_value=None, allow_none=True, help="Local working-cache override.")
        timeout = Float(default_value=None, allow_none=True, help="Per-scan estimation timeout in seconds.")

        def _estimate(self, session):
            """Estimate all selected scans, then attach and register outputs."""
            from pathlib import Path

            import nibabel as nib

            from .pglExternalTools import pglAFNI
            from .pglMRI import pglMRITransforms
            from .pglSession import pglSession

            self._motionRecord = None
            if not isinstance(session, pglSession) or session.mri is None:
                raise ValueError("Motion estimation requires a session with MRI scans.")

            for name in (self.outputName, self.correctedOutputName):
                if not name.strip(): raise ValueError("Product names must not be blank.")
            if self.outputName == self.correctedOutputName:
                raise ValueError("Motion and corrected-image products need different names.")

            existing = {
                record["name"]
                for entry in session.actionHistory
                for record in entry.actionProducts
                if record.get("recordStatus") == "committed"
            }
            requestedNames = {self.outputName}
            if not self.deferApply: requestedNames.add(self.correctedOutputName)
            if requestedNames & existing:
                raise ValueError(f"Output products already exist: {sorted(requestedNames & existing)}")

            # Reuse tested acquisition validation/selection. Geometry groups
            # do not imply cross-run alignment in this within-run action.
            plan = pglActionsMRI._prepareRunAlignment(
                session.mri,
                self.scanIndices,
                referenceRun="first",
            )

            sources = []
            for scanRecord in plan["scans"]:
                source = pglActionsMRI._resolveImageSource(
                    session,
                    {
                        "schemaVersion": 1,
                        "kind": "acquisition",
                        "scanUUID": scanRecord["scanUUID"],
                    },
                    cacheDirectory=self.cacheDirectory,
                )
                if source["volumeCount"] < 2 or self.referenceVolume >= source["volumeCount"]:
                    raise ValueError(f"Invalid motion reference/volume count for MRI scan {source['scanIndex']}.")
                sources.append(source)

            afni = pglAFNI(directory=self.afniDirectory)
            generated = []
            invocationIDs = []

            for source in sources:
                result = afni.volreg(
                    source["localPath"],
                    referenceVolume=self.referenceVolume,
                    twoPass=self.twoPass,
                    cacheDirectory=self.cacheDirectory,
                    timeout=self.timeout,
                    sourceStageID=source["sourceStageID"],
                    physicalSpaceID=f"acquisitionPhysical:{source['scanUUID']}",
                )
                if result is None: raise RuntimeError("AFNI motion estimation did not return a result.")

                details = result.provenance["invocation"]["result"]
                if details["sourceIdentity"] != source["sourceIdentity"]:
                    raise ValueError("AFNI input identity differs from the resolved acquisition.")

                image = nib.load(source["localPath"])
                units = image.header.get_xyzt_units()[1]
                secondsPerUnit = {"sec": 1.0, "msec": 0.001, "usec": 0.000001}.get(units)
                tr = None if secondsPerUnit is None else float(image.header.get_zooms()[3] * secondsPerUnit)
                if tr is not None and (not __import__("math").isfinite(tr) or tr <= 0): tr = None

                generated.append((source, result, details, tr))
                invocationIDs.append(result.invocationID)

            files = {}
            records = []
            executionID = self.actionHistory[-1].actionID

            for source, result, details, tr in generated:
                roles = {}
                for outputRole in ("matrices", "motionParameters"):
                    role = f"scan.{source['scanUUID']}.{outputRole}"
                    attachmentName = f"mri.{executionID}.{role}"
                    output = result.outputs[outputRole]

                    session.addDerivedFile(
                        attachmentName,
                        Path(output["localPath"]),
                        mediaType="text/plain",
                    )
                    attachment = session.derivedFiles[attachmentName]
                    if any(attachment[key] != output[key] for key in ("sha256", "size_bytes")):
                        raise ValueError("Motion output changed before registration.")

                    files[role] = attachmentName
                    roles[outputRole] = role

                records.append({
                    "scanIndex": source["scanIndex"],
                    "scanUUID": source["scanUUID"],
                    "sourceStageID": source["sourceStageID"],
                    "sourceIdentity": source["sourceIdentity"],
                    "volumeIndices": source["volumeIndices"],
                    "referenceVolumeIndex": self.referenceVolume,
                    "matricesRole": roles["matrices"],
                    "motionParametersRole": roles["motionParameters"],
                    "interpretation": details["interpretation"],
                    "timing": {"trSeconds": tr, "volumeTimesSeconds": None},
                })

            metadata = {
                "schemaVersion": 2,
                "operation": "withinRunMotion",
                "backend": "AFNI",
                "matrixConvention": {
                    "format": "AFNI aff12.1D",
                    "producerOption": "3dvolreg -1Dmatrix_save",
                    "mappingStatus": "interpreted",
                },
                "scans": records,
            }

            # Scientific validation belongs before action success, not in QC.
            artifactRecords = {
                role: {
                    "attachmentName": name,
                    **{key: session.derivedFiles[name][key] for key in ("sha256", "size_bytes")},
                }
                for role, name in files.items()
            }
            pglMRITransforms.validateMotionProduct({
                "kind": "rigidMotion",
                "metadata": metadata,
                "artifacts": artifactRecords,
            })

            self._motionRecord = self.registerProduct(
                self.outputName,
                "rigidMotion",
                session=session,
                files=files,
                metadata=metadata,
                invocationIDs=invocationIDs,
            )
            return session

        def _apply(self, session):
            """Apply this execution's exact motion result, not a named fallback."""
            if self._motionRecord is None:
                raise RuntimeError("No motion result was prepared by this execution.")

            pglActionsMRI._applyWithinRunMotion(
                self,
                session,
                self._motionRecord,
                outputName=self.correctedOutputName,
                cacheDirectory=self.cacheDirectory,
            )
            return session

        def _afterSuccess(self, session):
            """Show the committed motion product when requested."""
            if self.showMotion and self._motionRecord is not None:
                pglActionsMRI._showMotionProduct(session, self._motionRecord["productID"])
                
                
    ################################################################
    # Apply explicitly selected transforms
    ################################################################
    class mriApplyTransforms(pglAction):
        """Apply an explicitly selected transform product.

        Initial support is interpreted within-run AFNI rigid motion only.
        Cross-run composites, nonlinear warps, and derived source stages are
        rejected until their application adapters are implemented.

        Does not discover or apply unrelated pending products.
        """
        transforms = Instance(pglProductSelector, default_value=None, allow_none=True, help="Required explicit transform-product selector.")
        outputName = Unicode("motionCorrected", help="Name of the corrected-image product.")
        cacheDirectory = Unicode(default_value=None, allow_none=True, help="Local working-cache override.")

        def _run(self, session):
            """Resolve the declared product, verify it, and apply its mappings."""
            if self.transforms is None:
                raise ValueError("Supply transforms explicitly, for example pglProductSelector('motion').")
            if not self.outputName.strip():
                raise ValueError("outputName must not be blank.")

            product = self.transforms.resolve(session)
            if product is None: raise ValueError("Transform product could not be resolved.")
            if product.kind != "rigidMotion":
                raise ValueError(f"Application is not implemented for product kind {product.kind!r}.")

            pglActionsMRI._applyWithinRunMotion(
                self,
                session,
                product.record,
                outputName=self.outputName,
                cacheDirectory=self.cacheDirectory,
            )
            return session