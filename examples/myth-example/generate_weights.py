import random

w1_file = 'w1'
w2_file = 'w2'

inp_dim = 784
hidden_dim = 64
outp_dim = 10

with open(w1_file, 'w') as f:
    for i in range(inp_dim * hidden_dim):
        f.write('{}\n'.format(random.random() * 2 - 1))

with open(w2_file, 'w') as f:
    for i in range(hidden_dim * outp_dim):
        f.write('{}\n'.format(random.random() * 2 - 1))
