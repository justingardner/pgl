################################################################
#   filename: pglMessages.py
#    purpose: Centralized way to give warnings and errors
#         by: JLG
#       date: Jul 26, 2026
################################################################

#############
# Import
#############
import inspect
from IPython.display import display, HTML
import time
import threading


#################################################################
# warnings
#################################################################
class pglMessages:
    # keep track of oneTimeWarnings
    _oneTimeWarnings = set()
    
    @classmethod
    def message(cls, msg, callerNameDepth=2, verbose=True):
        if verbose:
            print(f"({cls.getCallerName(callerNameDepth)}) {cls.wrapText(msg)}")

    @classmethod
    def warning(cls, msg, level=2, callerNameDepth=2, verbose=True):
        if verbose:
            print(cls._formatMessage(msg,level,callerNameDepth))
    
    @classmethod
    def _formatMessage(cls, msg, level=2, callerNameDepth=3):
        msg = cls.wrapText(msg)
        if level == 0:
            msg = f"({cls.getCallerName(callerNameDepth)}) ⚠️ {msg} ⚠️"
        elif level == 1:
            msg = f"({cls.getCallerName(callerNameDepth)}) ❌ {msg} ❌"
        elif level == 2:
            msg = "❌"*80 + "\n" + f"({cls.getCallerName(callerNameDepth+1)}) {msg}\n" + "❌"*80
        return(msg)
    
    @staticmethod
    def wrapText(msg: str, lineLength: int = 80) -> str:
        """
        Split msg into lines, preferring to break at a space.
        If no space is found within lineLength characters, break at lineLength anyway.

        Args:
            msg: The string to wrap.
            lineLength: Max characters per line (default 80).

        Returns:
            The message with '\n' inserted as needed.
        """
        lines = []
        remaining = msg

        while len(remaining) > lineLength:
            # Look for the last space within the lineLength window
            breakAt = remaining.rfind(' ', 0, lineLength + 1)

            if breakAt == -1:
                # No space found within window; hard break at lineLength
                breakAt = lineLength
                lines.append(remaining[:breakAt])
                remaining = remaining[breakAt:]
            else:
                lines.append(remaining[:breakAt])
                remaining = remaining[breakAt + 1:]  # skip the space

        lines.append(remaining)
        return '\n'.join(lines) 
        
    @classmethod
    def oneTimeWarning(cls, msg, level=2):
        """
        Print a one-time warning message.

        Args:
            msg (str): The warning message to print.
            level (int, default=2): Severity of warning 
        """
        # check to see if we have already printed this warning
        if msg in cls._oneTimeWarnings:
            return
    
        # add the warning to the set of printed warnings
        cls._oneTimeWarnings.add(msg)
        
        # print the warning
        cls.warning(msg=msg, level=level, callerNameDepth=3)
       
    @classmethod
    def transientWarning(cls, msg, duration=5, level=2):
        """
        Display an HTML message that disappears after a specified duration.
        Only clears this specific message, not the whole cell.
            
        Args:
            message: The message to display
            duration: Time in seconds before the message disappears (default: None))
        """
        # if duration is set, then must use html
        if duration is not None:
            useHTML=True
            
        # Generate a unique display_id
        import uuid
        displayId = str(uuid.uuid4())
        
        # Display with the ID
        display(HTML(cls._formatMessage(msg,callerNameDepth=3,level=level)), display_id=displayId)
        
        # If no duration specified, we're done
        if duration is None:
            return
        
        def clearAfterDelay():
            time.sleep(duration)
            # Update using the display_id directly
            display(HTML(""), display_id=displayId, update=True)
        
        thread = threading.Thread(target=clearAfterDelay)
        thread.daemon = True
        thread.start()

    @classmethod
    def accessibilityWarning(cls):
        accessibilityWarningMessage = ("This app is not authorized for Accessibility input monitoring. No keyboard events will be detected!!" +
                    "  Go to System Settings → Privacy & Security → Accessibility and add this app." +
                    "  If you are running VS Code and it already has permissions granted, try running directly from a terminal with:" +
                    "  /Applications/Visual\\ Studio\\ Code.app/Contents/MacOS/Electron")
        cls.warning(accessibilityWarningMessage)
        
    @staticmethod
    def getCallerName(depth=2):
        """
        depth=1 -> caller of getCallerName()
        depth=2 -> caller of the caller
        etc.
        """
        frame = inspect.currentframe()

        for _ in range(depth):
            frame = frame.f_back

        functionName = frame.f_code.co_name

        runtimeClass = None
        definedClass = None

        # Find runtime class
        if "self" in frame.f_locals:
            obj = frame.f_locals["self"]
            runtimeClass = obj.__class__

        elif "cls" in frame.f_locals:
            obj = frame.f_locals["cls"]
            runtimeClass = obj

        # Find defining class using MRO
        if runtimeClass is not None:
            for cls in runtimeClass.mro():
                if functionName in cls.__dict__:
                    method = cls.__dict__[functionName]

                    if hasattr(method, "__code__"):
                        if method.__code__ is frame.f_code:
                            definedClass = cls
                            break

                    # handle classmethod / staticmethod
                    elif hasattr(method, "__func__"):
                        if method.__func__.__code__ is frame.f_code:
                            definedClass = cls
                            break

        # Format output
        if runtimeClass and definedClass:
            if runtimeClass == definedClass:
                return f"{runtimeClass.__name__}:{functionName}"
            else:
                return f"{runtimeClass.__name__}->{definedClass.__name__}:{functionName}"

        elif runtimeClass:
            return f"{runtimeClass.__name__}:{functionName}"

        else:
            moduleName = frame.f_globals.get("__name__", None)

            if functionName == "<module>":
                if moduleName and moduleName != "__main__":
                    return moduleName.split(".")[-1]
                else:
                    return "<jupyter:script>"
            else:
                if moduleName and moduleName != "__main__":
                    return f"{moduleName.split('.')[-1]}:{functionName}"
                else:
                    return functionName
