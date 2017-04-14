class CaptchaImageNotFoundException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__()

class CheckboxNotFoundException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__()

class InitialCheckboxNotFoundException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__()

class IFrameNotFoundException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__()

class SameCaptchaException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__()
