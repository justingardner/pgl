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
