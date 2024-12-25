import struct

SIZE_INT = 4
SIZE_DOUBLE = 8
SIZE_LONG_LONG = 8

class Converter:
    def __init__(self, partition_list, global_m, global_n, global_nnz, n_parts):
        self.partition_list = partition_list
        self.global_m = global_m
        self.global_n = global_n
        self.global_nnz = global_nnz
        self.n_parts = n_parts

    def writeBIN(self, filename):
        binfile = open(filename, "wb")
        #file = open("at.txt", "w")
        #write global m
        binfile.write(struct.pack('i', self.global_m))
        #file.write(str(self.global_m) + "\n")
        #print("Global M: ", self.global_m)
        #write global n
        binfile.write(struct.pack('i', self.global_n))
        #file.write(str(self.global_n) + "\n")
        #print("Global N: ", self.global_n)
        #write slocs
        slocs = [None] * self.n_parts
        slocs[0] = SIZE_INT+SIZE_INT + (SIZE_LONG_LONG*self.n_parts)
        binfile.write(struct.pack('q', slocs[0]))
        #file.write(str(slocs[0]) + "\n")
        #print("Slocs[0]: ", slocs[0])
        for p in range(1, self.n_parts):
            # p_size = local_m + local_nnz + little_m + ia + ja + jval + l2g_map
            psize = (3*SIZE_INT) + ((self.partition_list[p-1].local_m+1)*SIZE_INT) + \
                (self.partition_list[p-1].local_nnz*SIZE_INT) + (self.partition_list[p-1].local_nnz*SIZE_DOUBLE) + \
                (self.partition_list[p-1].local_m*SIZE_INT) 
            slocs[p] = slocs[p-1] + psize
            binfile.write(struct.pack('q', slocs[p]))
            #file.write(str(slocs[p]) + "\n")
            #print("Slocs[", p, "]: ", slocs[p])
        for part in self.partition_list:
            #write local m
            binfile.write(struct.pack('i', part.local_m))
            #file.write(str(part.local_m) + "\n")
            #print("Local M: ", part.local_m)
            #write local n
            binfile.write(struct.pack('i', part.local_nnz))
            #file.write(str(part.local_nnz) + "\n")
            #print("Local NNZ: ", part.local_nnz)
            #write little m
            binfile.write(struct.pack('i', part.little_m))
            #file.write(str(part.little_m) + "\n")
            #print("Little M: ", part.little_m)
            #write ia
            binfile.write(struct.pack('i'*len(part.ia), *(part.ia)))
            #file.write("IA: " + str(part.ia) + "\n")
            #print("IA: ", *(part.ia))
            #write ja
            binfile.write(struct.pack('i'*len(part.ja), *(part.ja)))
            #file.write("JA: " + str(part.ja) + "\n")
            #print("JA: ", *(part.ja))
            #write jval
            binfile.write(struct.pack('d'*len(part.jval), *(part.jval)))
            #file.write("JVAL: " + str(part.jval) + "\n")
            #print("JVAL: ", *(part.jval))
            #write l2g_map
            binfile.write(struct.pack('i'*len(part.l2g_map), *(part.l2g_map)))
            #file.write("L2G_MAP: " + str(part.l2g_map) + "\n")
            #print("L2G_MAP: ", *(part.l2g_map))
        #print("Slocs: ", slocs)
        binfile.close()
