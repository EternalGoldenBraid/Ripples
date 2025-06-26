class GuitarProfile:
    def __init__(self, open_strings: list[float], num_frets: int):
        self.open_strings = open_strings  # List of open string frequencies
        self.num_frets = num_frets

    def lowest_frequency(self) -> float:
        return min(self.open_strings)

    def highest_frequency(self) -> float:
        # Highest open string frequency plus frets
        highest_open = max(self.open_strings)
        semitones_up = self.num_frets
        return highest_open * (2 ** (semitones_up / 12))
