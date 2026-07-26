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

#################################################################
# warnings
#################################################################
class pglMessages:
    # keep track of oneTimeWarnings
    _oneTimeWarnings = set()
    
    @classmethod
    def warning(cls, msg, level=1, callerNameDepth=2):
        if level==1:
            print(f"({cls.getCallerName(callerNameDepth)}) ❌ {msg} ❌")
        elif level > 1:
            print("❌"*80)
            print(f"({cls.getCallerName(callerNameDepth)}) {msg}")
            print("❌"*80)   
            
    @classmethod
    def oneTimeWarning(cls, msg, level=2):
        """
        Print a one-time warning message.

        Args:
            msg (str): The warning message to print.
            level (int, default=2): Severity of warning 
        """
        # check to see if we have already printed this warning
        if msg in self._oneTimeWarnings:
            return
    
        # add the warning to the set of printed warnings
        self._oneTimeWarnings.add(msg)
        
        # print the warning
        cls.warning(msg=msg, level=level, callerNameDepth=3)
       
    @staticmethod 
    def getCallerName(depth=2):
        """
        depth=1 -> caller of getCaller()
        depth=2 -> caller of the caller
        etc.
        """
        frame = inspect.currentframe()

        for _ in range(depth):
            frame = frame.f_back

        functionName = frame.f_code.co_name

        if "self" in frame.f_locals:
            className = frame.f_locals["self"].__class__.__name__
        elif "cls" in frame.f_locals:
            className = frame.f_locals["cls"].__name__
        else:
            className = None

        if className:
            return f"{className}:{functionName}"
        else:
            return functionName

        
        
        
