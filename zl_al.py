class ZlAl():
    def __init__(self):
        self.zl = []
        self.al = []

    def add(self, zl_given, al_given):
        self.zl.append(zl_given)
        self.al.append(al_given)

    def getZl(self, index):
        return self.zl[index]
    def getAl(self, index):
        return self.al[index]

