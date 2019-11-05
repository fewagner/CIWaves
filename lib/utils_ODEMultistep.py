import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


# ------------------------------------------------------
# TRAIN FUNCTION
# ------------------------------------------------------

def train(args, model, criterion, train_loader, optimizer, epoch):
    model.train()
    print('\n')
    running_loss = 0
    t_start = time.time()
    for batch_idx, data in enumerate(train_loader):
        nr = data['k_n_r'][:, 1:].to(args.device)  # this is a tensore of size (batchsize, 10000)
        ni = data['n_i'].to(args.device)  # tensor size (batchsize, 10000)
        k = data['k_n_r'][:, 0].to(args.device)  # tensor size (batchsize)

        # for i in range(args.steps + 1, nr.size(1)):
        # ni_pred should be a tensor of length (i)
        # with torch.autograd.set_detect_anomaly(True):
        ni_pred = model(nr=nr,
                        iv=ni[:, :args.steps],
                        k=k,
                        stop=nr.size(1))
        # print('ni label: ', i, ni[:, :i])
        # print('ni pred: ', i, ni_pred)
        loss = criterion(ni_pred, ni)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # if i % 20 == 0:
        #    print('nodes done in this batch: ', i)
        print('PLOTTING INTERMEDIATE')
        plt.close('all')
        plt.plot(args.grid, ni_pred.detach().cpu().numpy()[0], label='ni_pred')
        plt.plot(args.grid, ni[0].detach().cpu().numpy(), label='ni_label')
        plt.legend()
        plt.savefig('plots/ODEMultistep_intermediate.pdf', dpi=300)
        # print('runtime for this batch: ', time.time() - t_start)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                running_loss / args.log_interval))
            running_loss = 0


# ------------------------------------------------------
# VALIDATION FUNCTION
# ------------------------------------------------------

def validation(args, model, criterion, validation_loader):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            nr = data['k_n_r'][:, 1:].to(args.device)
            ni = data['n_i'].to(args.device)
            k = data['k_n_r'][:, 0].to(args.device)

            # ni_pred should be a tensor of same length as (nr)
            ni_pred = model(nr=nr,
                            iv=ni[:, :args.steps],
                            k=k,
                            stop=nr.size(1))
            loss = criterion(ni_pred, ni)
            validation_loss += loss.item()
    validation_loss /= len(validation_loader)
    print('Validation set loss: {:.6f}'.format(validation_loss))

    return validation_loss


# ------------------------------------------------------
# MODEL WRAPPER FOR EVAL
# ------------------------------------------------------

class WrapperODEMultistep(nn.Module):

    def __init__(self, path_model, args):
        super(WrapperODEMultistep, self).__init__()
        self.realmodel = torch.load(path_model, map_location=args.device)
        self.args = args

    def forward(self, knr):
        nr = knr[:, 1:]
        k = knr[:, 0]

        out = self.realmodel(nr=nr,
                             iv=torch.zeros(self.args.batchsize, self.args.steps),
                             k=k,
                             stop=nr.size(1))

        return out

# ------------------------------------------------------
# MODEL
# ------------------------------------------------------

class ODEMultistep(nn.Module):

    def __init__(self, args, taylor=False):  # input k,h,nr,ni: 1,1,steps+1,steps
        super(ODEMultistep, self).__init__()
        if taylor:
            self.fnn1 = Derivative_FNN_taylor().to(args.device)
        else:
            self.fnn1 = Derivative_FNN_plain().to(args.device)
        self.fnn2 = Superpos_FNN(args.steps).to(args.device)  # hidden layer
        self.args = args

    def forward(self, nr, iv, k, stop):  # stop says how long the output should be
        ni_pred = []
        for idx in range(iv.size(1)):
            ni_pred.append(iv[:, idx])

        # this loops over all available nodes and calculates the next node
        for j in range(self.args.steps, stop):
            # print('j: ', j)
            # if j % 1000 == 0:
            #     print('calculating step: ', j)
            solutions = []
            # this calculates all previous derivatives
            for l in range(j - self.args.steps, j):
                # print(l)
                dnr = (nr[:, l + 1] - nr[:, l - 1]) / (2 * self.args.stepsize)
                # print('ni_pred[l]: ', ni_pred[l].size())
                derivative = self.fnn1(nr[:, l], ni_pred[l], dnr, k)
                solutions.append(derivative)
                # print(solutions)
            this_derivative = self.fnn2(solutions)
            next_ni = ni_pred[j - 1] + self.args.stepsize * this_derivative.view(-1)
            # print('der size: ', next_ni.size())
            # next_ni = next_ni.view(-1)
            ni_pred.append(next_ni)

        return torch.stack(ni_pred, dim=1)


class Derivative_FNN(nn.Module):

    def __init__(self):
        super(Derivative_FNN, self).__init__()
        self.linear1 = nn.Linear(2, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 1)

        # self.linear2 = nn.Linear(2, 1)

    def forward(self, nr, ni, dnr, k):
        t1 = nr / ni * dnr
        td = nr.pow(2)
        td = td - ni.pow(2)
        t2 = k * nr
        t2 = t2 * torch.sqrt(td)
        t1, t2 = t1.view(-1, 1), t2.view(-1, 1)

        input = torch.cat((t1, t2), dim=1)

        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out


class Derivative_FNN_plain(nn.Module):

    def __init__(self):
        super(Derivative_FNN_plain, self).__init__()
        self.linear1 = nn.Linear(8, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 1)

        # self.linear2 = nn.Linear(2, 1)

    def forward(self, nr, ni, dnr, k):
        nr = nr.view(-1, 1)
        ni = ni.view(-1, 1)
        dnr = dnr.view(-1, 1)
        k = k.view(-1, 1)

        input = torch.cat((nr, ni, dnr, k, nr ** 2, ni ** 2, 2 * k * nr, nr * dnr), dim=1)

        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out


class Derivative_FNN_taylor(nn.Module):

    def __init__(self):
        super(Derivative_FNN_taylor, self).__init__()
        self.linear1 = nn.Linear(13, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 1)

    def forward(self, nr, ni, dnr, k):
        # the usual suspects
        features = [nr, ni, dnr, k]

        # taylor series for 1/ni around +1
        features.append(nr * dnr)
        features.append(- (ni - 1) * nr * dnr)
        features.append((ni - 1) ** 2 * nr * dnr)
        # features.append(- (ni - 1) ** 3 * nr * dnr)
        # features.append((ni - 1) ** 4 * nr * dnr)

        # for elem in features[-5:]:
        #    elem[ni < 0] = 0

        # taylor series for 1/ni around -1
        features.append(- nr * dnr)
        features.append(- (ni + 1) * nr * dnr)
        features.append(- (ni + 1) ** 2 * nr * dnr)
        # features.append(- (ni + 1) ** 3 * nr * dnr)
        # features.append(- (ni + 1) ** 4 * nr * dnr)

        # for elem in features[-5:]:
        #    elem[ni > 0] = 0

        # taylor series for sqrt(1 - ni**2/nr*+2)
        features.append(2 * k * nr ** 2)
        features.append(2 * k * nr ** 2 * (-ni ** 2 / 2 / ni ** 2))
        features.append(2 * k * nr ** 2 * (-ni ** 4 / 8 / ni ** 4))
        # features.append(2 * k * nr ** 2 * (-ni ** 6 / 16 / ni ** 6))
        # features.append(2 * k * nr ** 2 * (-5 * ni ** 8 / 128 / ni ** 8))

        features = torch.stack(features, dim=1)

        # print(features)

        out = F.relu(self.linear1(features))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)

        # print(out)

        return out


class Superpos_FNN(nn.Module):

    def __init__(self, steps):  # input k,h,nr,ni: 1,1,steps+1,steps
        super(Superpos_FNN, self).__init__()
        self.linear = nn.Linear(steps, 1)  # hidden layer

    def forward(self, solutions):
        input = solutions[0]
        for val in solutions[1:]:
            input = torch.cat((input, val), dim=1)
        return self.linear(input)
