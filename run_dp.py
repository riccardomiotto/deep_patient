from scipy import sparse
import deep_patient as dp
import numpy as np

# create fake data
rndm = np.random.randint(low=0, high=25, size=[100, 50])
data = sparse.csc_matrix(rndm, dtype=float)

# initiate the model
nhidden = 10
nlayer = 3
sda = dp.SDA(data.shape[1],
             nhidden=nhidden,
             nlayer=nlayer,
             param={
    'epochs': 10,
    'batch_size': 5,
    'corrupt_lvl': 0.05
})

# train the model
sda.train(data)

# apply the mode
deep_repr = sda.apply(data)

print '\nfinal representation\n'
print deep_repr
