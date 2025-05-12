""" Caleb Kim
    5/11/2025
    CS 4348.501
"""

import sys # for command line arguments
import os # for file operations
import struct # for packing and unpacking binary data
from collections import OrderedDict # for maintaining order of keys in dictionary

"""
    Constants for the index file which will be divided into blocks of 512 bytes:
    * Each node of the btree will fit into one 512 byte block and the file hedaer will use the entire first block.
    * New nodes will be appended to the end of the file.
    - The size of the header information and a node will be smaller than a block size.
    - The remaining space in the block will remain unused.
    * All numbers stored in the file should be stored as 8-byte integers with the big endian byte order.
    - (The most significant byte is stored at the lowest memory address.)
    - Python to_bytes() function will be used for this operation.
    * Since there's no delete operation, we don't need to worry about deleting nodes. 
"""
BLOCK_SIZE = 512
MAGIC = b"4348PRJ3" # 8-byte magic number that's actually a sequence of ASCII values
T = 10 # minimum degree of the B-Tree
MAX_KEYS = 2 * T - 1 # maximum number of keys in a node
MAX_CHILDREN = 2 * T # maximum number of children in a node
CACHE_SIZE = 3 # max nodes in memory

"""
    File management class that reads files and performs error handling when necessary
"""
class FileManager:
    def __init__(self, path): # constructor
          self.path = path
          self.fd = None
          self.root_id = 0
          self.next_block_id = 1
          self.cache = OrderedDict () # LRU cache

    def open(self, create=False):
     mode = 'r+b' if not create else 'w+b' # open file in read/write mode
     self.fd = open(self.path, mode)
     
     # write header block
     if create:
        buf = bytearray(BLOCK_SIZE)
        buf[0:8] = MAGIC
        struct.pack_into('>Q', buf, 8, 0) # root id
        struct.pack_into('>Q', buf, 16, 1) # next block id
        self.fd.write(buf)
        self.fd.flush() # write to disk
    
     # read header block
     else:
         self.fd.seek(0) # move to the beginning of the file
         hdr = self.fd.read(BLOCK_SIZE)
         if hdr[0:8] != MAGIC:
                raise ValueError("Invalid index file format")
         self.root_id = struct.unpack('>Q', hdr, 8)[0] # read root id
         self.next_block_id = struct.unpack('>Q', hdr, 16)[0]
     
    def sync_header(self): # write the root id and next block id to the header block
        buf = bytearray(BLOCK_SIZE)
        buf[0:8] = MAGIC
        struct.pack_into('>Q', buf, 8, self.root_id)
        struct.pack_into('>Q', buf, 16, self.next_block_id)
        self.fd.seek(0)
        self.fd.write(buf)
        self.fd.flush()

    def allocate_block(self): # allocate a new block in the file
        bid = self.next_block_id
        self.next_block_id += 1
        self.sync_header()
        return bid
    
    def read_node(self, block_id):
        # LRU cache lookup
        if block_id in self.cache:
            # move the accessed node to the end of the cache
            node = self.cache.pop(block_id)
            self.cache[block_id] = node
            return node
        
        # read from disk
        self.fd.seek(block_id * BLOCK_SIZE)
        data = self.fd.read(BLOCK_SIZE)
        node = BTreeNode.from_bytes(data)
        self.cache[block_id] = node

        if len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)
        return node
    
    def write_node(self, node):
        data = node.to_bytes()
        self.fd.seek(node.block_id * BLOCK_SIZE)
        self.fd.write(data)
        self.fd.flush()

        # update the cache
        self.cache[node.block_id] = node
        if len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)

    def close(self):
        if self.fd:
            self.fd.close()
            self.fd = None
            self.cache.clear()

"""
    B-Tree node class that represents a node in the B-Tree:
    * Handles the operations on a singular node as compared to the entire tree which will help when performing operations for the BTree class.
    * Each node will be stored in a block of 512 bytes.
"""
class BTreeNode:
    HEADER_FMT = '>QQQ' # block_id, parent_id, num_keys

    def __init__(self, block_id, parent_id=0, keys=None, values=None, children=None):
        self.block_id = block_id
        self.parent_id = parent_id
        self.keys = keys or []
        self.num_children = children or []
        self.values = values or []

    @classmethod
    def from_bytes(cls, data): # convert bytes to node
        bid, pid, nkeys = struct.unpack_from(cls.HEADER_FMT, data, 0)
        offset = struct.calcsize(cls.HEADER_FMT)
        keys = list(struct.unpack_from('>Q'*MAX_KEYS, data, offset))
        offset += 8*MAX_KEYS
        vals = list(struct.unpack_from('>Q'*MAX_KEYS, data, offset))
        offset += 8*MAX_KEYS
        ch = list(struct.unpack_from('>Q'*MAX_CHILDREN, data, offset))

        # trim the keys and values to the number of keys
        keys = keys[:nkeys]
        vals = vals[:nkeys]
        ch = ch[:nkeys+1]
        return cls(bid, pid, keys, vals, ch)
    
    def to_bytes(self): # convert the node to bytes
        buf = bytearray(BLOCK_SIZE)
        struct.pack_into(self.HEADER_FMT, buf, 0, self.block_id, self.parent_id, len(self.keys))
        offset = struct.calcsize(self.HEADER_FMT)

        # pack the keys
        for i in range(MAX_KEYS):
            struct.pack_into('>Q', buf, offset, self.keys[i] if i < len(self.keys) else 0)
            offset += 8

        # pack the values
        for i in range(MAX_KEYS):
            struct.pack_into('>Q', buf, offset, self.values[i] if i < len(self.values) else 0)
            offset += 8

        # pack the children
        for i in range(MAX_CHILDREN):
            struct.pack_into('>Q', buf, offset, self.num_children[i] if i < len(self.num_children) else 0)
            offset += 8
        return bytes(buf)
    
    def is_leaf(self):
        return len(self.num_children) == 0

"""
    B-Tree class:
    * This class will handle the operations on the B-Tree as a whole.
    * create
    - Create a new index file.
    - The first argument after "create" is assumed to be the name of the index file.
    - If that file already exists, fail with an error message.
    - (The file should remain unmodified.)
    - (Example: project3 create test.idx)
    * insert
    - The first arugment after "insert" is assumed to be the name of the index file.
    - If the file doesn't exist or isn't a valid index file, exit with an error.
    - The next two arguments are the key and value, respectively and should be converted into unsigned integers which will them be inserted into the B-Tree.
    - (Example: project3 insert test.idx 15 100)
    * search
    - The first argument after "search" is assumed to be the name of the index file.
    - If the file doesn't exist or isn't a valid index file, exit with an error.
    - The next argument is assumed to be a key which is converted into an unsigned integer.
    - Search the index for the key and if it exists, print the key / value pair.
    - If the key doesn't exist, print an error message.
    - (Example: project3 search test.idx 15)
    * load
    - The first argument after "load" is assumed to be the name of the index file.
    - If the file doesn't exist or isn't a valid index file, exit with an error.
    - The next argument is assumed to be the name of a csv file, if the file doesn't exist then exit with an error message.
    - Each line of the file is a comma separated key / value pair.
    - Read the file, inserting each pair as above with the insert command.
    - (Example: project3 load test.idx input.csv)
    * print
    - The first argument after "print" is assumed to be the name of the index file.
    - If the file doesn't exist or isn't a valid index file, exit with an error.
    - Assuming the file exists, print every key / value pair in the index to standard output.
    - (Example: project3 print test.idx)
    * extract
    - The first argument after "extract" is assumed to be the name of the index file.
    - If the file doesn't exist or isn't a valid index file, exit with an error.
    - The second argument is the name of the file and if it exists, exit with an error.
    - (The file should remain unmodified.)
    - Save every key / value pair in the index as comma separated pairs to the file.
    - (Example: project3 extract test.idx output.csv)
"""
class BTree:
    def __init__(self, bfile: FileManager):
        self.bfile = bfile

    def search(self, block_id, key):
        if block_id == 0:
            return None
        node = self.bfile.read_node(block_id)
        i = 0

        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key > node.keys[i]:
            return node.values[i]
        if node.is_leaf():
            return None
        return self.search(node.num_children[i], key)
    
    def traverse(self, block_id, visit):
        if block_id == 0:
            return
        node = self.bfile.read_node(block_id)
        for i, k in enumerate(node.keys):
            if not node.is_leaf():
                self.traverse(node.children[i], visit)
            visit(k, node.values[i])
        if not node.is_leaf():
            self.traverse(node.children[len(node.keys)], visit)

    def collect(self):
        pairs = []
        self.traverse(self.bfile.root_id, lambda k,v: pairs.append((k, v)))
        return pairs
    
    def insert(self, key, value):
        # empty tree
        if self.bfile.root_id == 0:
            bid = self.bfile.allocate_block()
            root = BTreeNode(bid)
            root.keys, root.values = [key], [value]
            self.bfile.write_node(root)
            self.bfile.root-id = bid
            self.bfile.syn_header()
            return
        
        root = self.bfile.read_node(self.bfile.root_id)
        if len(root.keys) == MAX_KEYS:
            # split the root
            new_bid = self.bfile.allocate_block()
            new_root = BTreeNode(new_bid, parent_id=0, children=[root.block_id])
            root.parent_id = new_bid
            self.bfile.write_node(root)
            self.split_child(new_root, 0)
            self.bfile.root_id = new_bid
            self.bfile.sync_header()
            self.insert_nonfull(new_root, key, value)

        else:
            self.insert_nonfull(root, key, value)

    def split_child(self, parent, i):
        t = T
        y = self.bfile.read_node(parent.children[i])
        z_bid = self.bfile.allocate_block()
        z = BTreeNode(z_bid, parent_id=parent.block_id)
        mid_key = y.keys[t-1]
        mid_val = y.values[t-1]

        # split the keys and values
        z.keys = y.keys[:t]
        z.values = y.values[:t]
        y.keys = y.keys[:t-1]
        y.values = y.values[:t-1]

        # split children if not leaf
        if not y.is_leaf():
            z.children = y.children[t:]
            y.children = y.children[:t]
            for c in z.children:
                if c != 0:
                    child = self.bfile.read_node(c)
                    child.parent_id = z_bid
                    self.bfile.write_node(child)

        # insert into parent
        parent.keys.insert(i, mid_key)
        parent.values.insert(i, mid_val)
        parent.children.insert(i+1, z_bid)

        # write the new nodes to disk
        self.bfile.write_node(y)
        self.bfile.write_node(z)
        self.bfile.write_node(parent)

    def insert_nonfull(self, node, key, value):
        if node.is_leaf():
            # find the position to insert the new key
            i = len(node.keys)-1
            node.keys.append(0)
            node.values.append(0)

            while i >= 0 and key < node.keys[i]:
                node.keys[i+1] = node.keys[i]
                node.values[i+1] = node.values[i]
                i -= 1
            node.keys[i+1] = key
            node.values[i+1] = value
            self.bfile.write_node(node)

        else:
            i = len(node.keys)-1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            child = self.bfile.read_node(node.children[i])
            if len(child.keys) == MAX_KEYS:
                self.split_child(node, i)
                if key > node.keys[i]:
                    i += 1
                child = self.bfile.read_node(node.children[i])
            self.insert_nonfull(child, key, value)

"""
    main function which will be used to test the B-Tree implementation.
"""
def print_menu():
    print("\nCommands:")
    print("  create   - create a new index file")
    print("  open     - open existing index file")
    print("  insert   - insert a key-value pair")
    print("  search   - search for a key")
    print("  load     - bulk load from CSV")
    print("  print    - print all entries")
    print("  extract  - extract entries to CSV")
    print("  quit     - exit")

def main():
    bfile = None
    btree = None
    print_menu()

    while True:
        cmd = input("Enter command: ").strip().lower() # get user input
        if cmd == 'quit':
            break
        elif cmd == 'create':
            fname = input("File name: ").strip()
            if os.path.exists(fname) and input("Overwrite? (y/n): ").strip().lower() != 'y':
                print("File already exists. Use 'open' to open the file.")
                continue

            bfile = FileManager(fname)
            bfile.open(create=True)
            btree = BTree(bfile)
            print(f"Created index file: {fname}.")

        elif cmd == 'open':
            fname = input("File name: ").strip()
            if not os.path.exists(fname):
                print(f"File {fname} does not exist.")
                continue

            bfile = FileManager(fname)
            try:
                bfile.open(create=False)
            except Exception as e:
                print(f"Error: {e}")
                bfile = None
                continue
            tree = BTree(bfile)
            print(f"Opened index file: {fname}.")

        else:
            if not btree:
                print("No index file opened. Use 'create' or 'open' first.")
                continue
            if cmd == 'insert':
                k = int(input("Key: "))
                v = int(input("Value: "))
                btree.insert(k, v)
            
            elif cmd == 'search':
                k = int(input("Key: "))
                res = btree.search(bfile.root_id, k)
                print(res if res is not None else "Not found.")

            elif cmd == 'load':
                csvf = input("CSV file name: ").strip()
                if not os.path.exists(csvf):
                    print(f"File {csvf} does not exist.")
                    continue
                with open(csvf) as f:
                    for line in f:
                        parts=line.strip().split(',')
                        if len(parts) == 2:
                            btree.insert(int(parts[0]), int(parts[1]))
            
            elif cmd == 'print':
                for k, v in btree.collect():
                    print(f"{k}, {v}")
            
            elif cmd == 'extract':
                out = input("Output CSV file name: ").strip()
                if os.path.exists(out) and input("Overwrite? (y/n): ").strip().lower() != 'y':
                    continue
                with open(out, 'w') as of:
                    for k,v in btree.collect():
                        of.write(f"{k},{v}\n")
                print(f"Extracted to {out}.")
            
            else: 
                print(f"Unknown command: {cmd}")

    if bfile:
        bfile.close()

if __name__ == "__main__":
    main()