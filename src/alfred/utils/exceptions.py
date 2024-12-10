class NotEnoughDataError(Exception):
    """Custom exception to indicate not enough data is available."""
    def __init__(self, message="Not enough data provided"):
        self.message = message
        super().__init__(self.message)