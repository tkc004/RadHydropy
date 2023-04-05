class Par():
    def __init__(self,params) -> None:
            for key, value in params.items():
                setattr(self, key, value)        

        