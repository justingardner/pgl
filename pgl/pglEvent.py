################################################################
#   filename: pglEvent.py
#    purpose: Events for recording timestamps of keyboard, mouse
#             digio and other events
#         by: JLG
#       date: July 28, 2025
################################################################

#############
# Import modules
#############
from .pglSerialize import pglSerialize
from dataclasses import dataclass, field
from typing import List, Optional

#################################################################
# Parent class for events
#################################################################
class pglEvent(pglSerialize):
    """
    Parent class for all pglEvent types
    """
    def __init__(self, type="pglEvent", **kwargs):
        """
        Initialize the pglEvent instance.
        
        Args:
            type (str): The type/category of this event
        """
        self.type = type
        
        # if the subclass event has set annotations and there
        # are passed i n kwargs, then set those fields
        fields = vars(self.__class__).get('__annotations__', {})
        # for every field in annotations
        for f in fields:
            # if it ia an input field
            if f in kwargs:
                # set the attribute
                setattr(self, f, kwargs[f])
    
    # keep a registry of pglEvent subclasses             
    _registry = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Automatically register every subclass
        pglEvent._registry[cls.__name__] = cls
        
    @classmethod
    def getClass(cls, className):
        '''
        get sub classes from classNames as created by registry above
        '''
        if className not in cls._registry:
            raise ValueError(
                f"(pglEvent:getClass) Unknown event class '{className}'"
            )
        return cls._registry[className]
    
    def __repr__(self):
        if hasattr(self, 'type'):
            return f"<pglEvent type={self.type}>"
        else:
            return f"<pglEvent type={self.__class__.__name__}>"

    def print(self):
        """
        Print the details of the pglEvent instance.
        """
        if hasattr(self, 'type'):
            print(f"(pglEvent) Type: {self.type}")
        else:
            print(f"<pglEvent type={self.__class__.__name__}>")
        for name, value in vars(self).items():
            print(f"  {name}: {value}")
#############
# pglEvents - Event container using dataclass
#############
@dataclass
class pglEvents():
    """
    Container for managing multiple pglEvent instances.

    This class provides methods for handling events related to keyboard, 
    mouse, and other device interactions.
    
    Attributes:
        events (List[pglEvent]): FIFO queue of events
    """
    events: List[pglEvent] = field(default_factory=list)
    
    ################################################################
    # Add events to the event list
    ################################################################
    def eventsAdd(self, events):
        """
        Add events to the event list.

        Args:
            events (pglEvent or list): Event(s) to add.

        Returns:
            None
        """
        if isinstance(events, list):
            self.events.extend(events)
        else:
            self.events.append(events)

    ################################################################
    # Get an event
    ################################################################
    def eventsGet(self) -> Optional[pglEvent]:
        """
        Get and remove the next event from the queue (FIFO).

        Returns:
            pglEvent or None: The next event, or None if queue is empty.
        """ 
        if self.events:
            return self.events.pop(0)
        return None
    
    ################################################################
    # eventsWaitFor
    ################################################################
    def eventsWaitFor(self, waitForList):
        """
        Wait for specific events to occur.

        Args:
            waitForList (list): A list of event types/IDs to wait for.

        Returns:
            pglEvent: The first matching event.
        """
        while True:
            events = self.poll()
            if events:
                for event in events:
                    if hasattr(event, 'id') and event.id in waitForList:
                        return event
            self.waitSecs(0.01)
            
###################################
# Keyboard event
###################################
class pglEventKeyboard(pglEvent):
    """
    Represents a keyboard event for the pglKeyboard device.

    """

    def __init__(self, keyChar=None, keyCode=None, timestamp=None, key=None, shift=False, ctrl=False, alt=False, cmd=False, eventType=None):
        '''
        Initialize the pglEventKeyboard instance.
        Args:
            keyChar(str): The key that was pressed. If not passed in, but keyCode is then is derived from keyCode
            keyCode (int): The key code of the pressed key. If not passed in, but keyChar is then is derived from keyChar
            key (Key): The key object. Can be omitted
            timestamp (double): The device time. 
            shift (bool): Whether the shift key was held down. Default False
            ctrl (bool): Whether the ctrl key was held down. Default False
            alt (bool): Whether the alt key was held down. Default False
            cmd (bool): Whether the cmd key was held down. Default False
            eventType (str): The type of event ('keydown' or 'keyup'). Defaults to keydown
        Returns:
            None
        '''
        super().__init__("keyboard")

        # if keyChar is not passed in but keyCode, is then dervie it from keyCode
        if keyChar is None and keyCode is not None:
            self.keyChar = keyCodeToChar(keyCode)
        else:
            self.keyChar = keyChar
        # get keyCode (either derive from keyChar or passed in)
        if keyCode is None and keyChar is not None:
            self.keyCode = charToKeyCode(keyChar)
        else:
            self.keyCode = keyCode
        self.key = key
        self.timestamp = timestamp
        self.shift = shift
        self.ctrl = ctrl
        self.alt = alt
        self.cmd = cmd
        # default to keydown events
        if eventType is not None:
            self.eventType = eventType
        else:
            self.eventType = 'keydown'
    def __repr__(self):
        '''
        Return a string representation of the pglEventKeyboard instance.
        Returns:
            str: String representation of the instance.
        '''
        modifierStr = ""
        if self.shift: modifierStr += "Shift "
        if self.ctrl: modifierStr += "Ctrl "
        if self.alt: modifierStr += "Alt "
        if self.cmd: modifierStr += "Cmd "
        return f"(pglEventKeyboard) Key: {self.keyChar}, KeyCode: {self.keyCode}, Timestamp: {self.timestamp}, Modifiers: {modifierStr.strip()}, Event Type: {self.eventType}"

#################################################################
# Digital input event
#################################################################
class pglEventDigitalInput(pglEvent):
    """An acquired digital edge with sample timing and a host-clock timestamp."""

    def __init__(
        self, channel=None, eventType=None, timestamp=None,
        sampleIndex=None, scanRate=None, deviceType=None,
        timestampMethod="streamEstimate", *,
        labJackTimestamp=None, clockMappingId=None,
        deviceTimestampMethod=None,
    ):
        """
        Args:
            channel (str): Physical input name, such as FIO0.
            eventType (str): rising or falling.
            timestamp (float): Estimated edge time in the pglTimestamp clock domain.
            sampleIndex (int): Zero-based scan index; acquisition metadata defines scope.
            scanRate (float): Actual scan rate returned by the device.
            deviceType (str): Device that acquired the event.
            timestampMethod (str): How the host-clock timestamp was obtained.
            labJackTimestamp (float or None): Unwrapped device coordinate in
                nominal core-timer seconds. Not calendar time.
            clockMappingId (str or None): eventId of the mapping record used
                to produce timestamp.
            deviceTimestampMethod (str or None): Provenance of the device
                coordinate, distinguishing sample-derived timing from capture.

        The timestamp identifies the first sample showing the new state.
        Sample resolution and absolute host-clock alignment are separate.

        Calibration evidence and uncertainty are retained in referenced
        calibration/mapping events, not duplicated on every edge.

        Post-run correction preserves liveTimestamp and liveClockMappingId
        on first correction. They are not populated by this constructor.
        """
        super().__init__(type="digitalInput")

        # Physical source and edge polarity.
        self.channel = channel
        self.eventType = eventType
        self.deviceType = deviceType

        # Preserve sample timing independently of the estimated host-clock mapping.
        self.sampleIndex = sampleIndex
        self.scanRate = scanRate
        self.sampleTime = sampleIndex / scanRate if sampleIndex is not None and scanRate is not None else None
        self.timestamp = timestamp
        self.timestampMethod = timestampMethod

        # Compact provenance; None explicitly means unavailable.
        self.labJackTimestamp = labJackTimestamp
        self.clockMappingId = clockMappingId
        self.deviceTimestampMethod = deviceTimestampMethod

    def __repr__(self):
        return f"<pglEventDigitalInput channel={self.channel!r} edge={self.eventType!r} timestamp={self.timestamp} sampleIndex={self.sampleIndex}>"
    
class pglEventLabJackClockCalibration(pglEvent):
    """
    Retained device/host clock observations from one calibration collection.

    timestamp is the host collection-completion time, not a device timestamp.
    Observations and diagnostics are independent snapshots.

    Calibration IDs use eventId. Failed collections remain valid records,
    but cannot by themselves authorize a timestamp conversion.
    """

    def __init__(
        self, *, connectionId, deviceIdentity, phase,
        collectionStartedTimestamp, collectionCompletedTimestamp,
        requestedDuration, observations, status, diagnostics=None,
        eventId=None,
    ):
        from copy import deepcopy
        from math import isfinite
        from uuid import uuid4

        if not isinstance(connectionId, str) or not connectionId:
            raise ValueError("connectionId must be a nonempty string.")
        if not isinstance(deviceIdentity, dict):
            raise TypeError("deviceIdentity must be a dictionary.")
        if phase not in {"startup", "beginning", "intermediate", "ending", "characterization"}:
            raise ValueError("Unknown clock-calibration phase.")
        if status not in {"collected", "failed", "cancelled"}:
            raise ValueError("Calibration status must be collected, failed, or cancelled.")
        if not isinstance(observations, list):
            raise TypeError("observations must be a list.")
        if any(not isinstance(observation, dict) for observation in observations):
            raise TypeError("Each observation must be a dictionary.")
        if diagnostics is not None and not isinstance(diagnostics, dict):
            raise TypeError("diagnostics must be a dictionary or None.")

        for name, value in (
            ("collectionStartedTimestamp", collectionStartedTimestamp),
            ("collectionCompletedTimestamp", collectionCompletedTimestamp),
            ("requestedDuration", requestedDuration),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number.")

        if requestedDuration <= 0:
            raise ValueError("requestedDuration must be positive.")
        if collectionCompletedTimestamp < collectionStartedTimestamp:
            raise ValueError("Collection completion precedes collection start.")

        if eventId is None: eventId = str(uuid4())
        if not isinstance(eventId, str) or not eventId:
            raise ValueError("eventId must be a nonempty string.")

        super().__init__(type="labJackClockCalibration")

        self.eventId = eventId
        self.connectionId = connectionId
        self.deviceType = "LabJack"
        self.deviceIdentity = deepcopy(deviceIdentity)
        self.phase = phase

        self.timestamp = float(collectionCompletedTimestamp)
        self.timestampMethod = "hostClockCalibrationCollectionCompletion"
        self.labJackTimestamp = None
        self.clockMappingId = None

        self.collectionStartedTimestamp = float(collectionStartedTimestamp)
        self.collectionCompletedTimestamp = float(collectionCompletedTimestamp)
        self.requestedDuration = float(requestedDuration)
        self.actualDuration = float(
            collectionCompletedTimestamp - collectionStartedTimestamp
        )

        self.observations = deepcopy(observations)
        self.status = status
        self.diagnostics = deepcopy(diagnostics) if diagnostics is not None else {}
        self.schemaVersion = 1


class pglEventLabJackClockMapping(pglEvent):
    """
    A retained conversion result derived from identified calibration events.

    eventId is the mapping ID referenced by acquisition events.
    timestamp is host mapping-creation time.

    This is a persistence record, not a clock fitter or validity authority.
    The driver must establish validity before emitting an applicable mapping.
    """

    def __init__(
        self, *, connectionId, calibrationEventIds, createdTimestamp,
        purpose, status, method, methodVersion, parameters=None,
        uncertainty=None, assumptions=None, diagnostics=None, eventId=None,
    ):
        from copy import deepcopy
        from math import isfinite
        from uuid import uuid4

        if not isinstance(connectionId, str) or not connectionId:
            raise ValueError("connectionId must be a nonempty string.")
        if (
            not isinstance(calibrationEventIds, (list, tuple))
            or not calibrationEventIds
            or any(not isinstance(value, str) or not value for value in calibrationEventIds)
            or len(set(calibrationEventIds)) != len(calibrationEventIds)
        ):
            raise ValueError("calibrationEventIds must contain distinct nonempty IDs.")
        if purpose not in {"live", "postRun", "characterization"}:
            raise ValueError("Unknown clock-mapping purpose.")
        if status not in {"validConditional", "rejected"}:
            raise ValueError("Mapping status must be validConditional or rejected.")
        if not isinstance(method, str) or not method:
            raise ValueError("method must be a nonempty string.")
        if not isinstance(methodVersion, str) or not methodVersion:
            raise ValueError("methodVersion must be a nonempty string.")
        if (
            isinstance(createdTimestamp, bool)
            or not isinstance(createdTimestamp, (int, float))
            or not isfinite(createdTimestamp)
        ):
            raise ValueError("createdTimestamp must be finite.")

        for name, value in (
            ("parameters", parameters),
            ("uncertainty", uncertainty),
            ("diagnostics", diagnostics),
        ):
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be a dictionary or None.")

        if assumptions is not None and (
            not isinstance(assumptions, list)
            or any(not isinstance(value, str) for value in assumptions)
        ):
            raise TypeError("assumptions must be a list of strings or None.")

        if status == "validConditional" and (
            parameters is None or uncertainty is None
        ):
            raise ValueError("An applicable mapping requires parameters and uncertainty metadata.")

        if eventId is None: eventId = str(uuid4())
        if not isinstance(eventId, str) or not eventId:
            raise ValueError("eventId must be a nonempty string.")

        super().__init__(type="labJackClockMapping")

        self.eventId = eventId
        self.connectionId = connectionId
        self.deviceType = "LabJack"
        self.calibrationEventIds = list(calibrationEventIds)

        self.timestamp = float(createdTimestamp)
        self.timestampMethod = "hostClockMappingCreation"
        self.labJackTimestamp = None
        self.clockMappingId = None

        self.purpose = purpose
        self.status = status
        self.method = method
        self.methodVersion = methodVersion
        self.parameters = deepcopy(parameters)
        self.uncertainty = deepcopy(uncertainty)
        self.assumptions = deepcopy(assumptions) if assumptions is not None else []
        self.diagnostics = deepcopy(diagnostics) if diagnostics is not None else {}
        self.schemaVersion = 1