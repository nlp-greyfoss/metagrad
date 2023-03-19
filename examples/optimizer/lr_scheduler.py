import matplotlib.pyplot as plt

import metagrad

model = metagrad.nn.Linear(2, 1)
optimizer = metagrad.optim.SGD(model.parameters(), lr=100)
# scheduler = metagrad.optim.ExponentialLR(optimizer, gamma=0.1, verbose=True)
# scheduler = metagrad.optim.StepLR(optimizer, step_size=2, gamma=0.1, verbose=True)
# scheduler = metagrad.optim.MultiStepLR(optimizer, milestones=[6, 8, 9], gamma=0.1, verbose=True)
#lambda1 = lambda epoch: 0.65 ** epoch
# scheduler = metagrad.optim.LambdaLR(optimizer, lr_lambda=lambda1, verbose=True)
scheduler = metagrad.optim.NoamLR(optimizer, 512)

lrs = []

for i in range(20000):
    optimizer.zero_grad()
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(20000), lrs)
plt.show()
