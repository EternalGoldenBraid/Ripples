class Note:
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
             'F#', 'G', 'G#', 'A', 'A#', 'B']
    name_to_index = {name: i for i, name in enumerate(names)}

    def __init__(self, name_or_index):
        if isinstance(name_or_index, int):
            self.index = name_or_index % 12
        elif isinstance(name_or_index, str):
            self.index = self.name_to_index[name_or_index]
        else:
            raise ValueError("Must be int or str")

    def __add__(self, steps):
        return Note((self.index + steps) % 12)

    def __str__(self):
        return self.names[self.index]

    def __repr__(self):
        return f"Note('{self.__str__()}')"

    def __eq__(self, other):
        return isinstance(other, Note) and self.index == other.index


class Scale:
    def __init__(self, root, steps):
        self.root = root
        self.steps = steps
        self.notes = self._build_notes()

    def _build_notes(self):
        total = 0
        notes = []
        for step in self.steps:
            notes.append(self.root + total)
            total += step
        return notes

    def __add__(self, shift):
        new_root = self.root + shift
        return Scale(new_root, self.steps)

    def __iter__(self):
        return iter(self.notes)

    def __repr__(self):
        return f"Scale(root={self.root}, steps={self.steps})"
