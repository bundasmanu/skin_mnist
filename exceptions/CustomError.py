
class ErrorCreationModel(Exception):

    def __init__(self, msg):
        if msg is None:
            msg = "\nNon defined msg"
        super(ErrorCreationModel, self).__init__(msg)