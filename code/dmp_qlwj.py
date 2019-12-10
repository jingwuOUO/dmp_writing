import numpy as np
import math

class DynamicMP:
    alpha = 0.4
    beta = 0.8
    # self.m = 300
    freq = 120
    time_delta = 1 / freq
    ci = 0


    def __init__(self, m, y_all_i, dy_all_i, ddy0_all_i, lenth, shape=0, linear=0):

        # self.ci = i/8
        self.m = m
        self.shape = shape
        self.linear = linear
        self.y = y_all_i
        self.dy = dy_all_i
        self.ddy0 = ddy0_all_i
        self.lenth = lenth
        self.x = np.zeros((self.lenth, 1), dtype=np.float)
        self.s = np.zeros((self.lenth,1), dtype=np.float)
        self.Phi = np.zeros((self.lenth, self.lenth), dtype=np.float)
        self.f = np.zeros((self.lenth, 1), dtype=np.float)
        self.w = np.zeros((self.m, 1), dtype=np.float)


    def getX(self):
        for i in range(self.lenth):
            if i == 0:
                self.x[i] = 1
            else:
                if self.linear == 1:
                    self.x[i] = self.x[i-1] - self.time_delta*self.x[i-1]
                elif self.linear == 0:
                    self.x[i] = 1 - i/self.lenth

    def getS(self):
        tmp = (self.alpha*self.beta)
        for i in range(self.lenth):
            self.s[i] = self.x[i] * tmp
        # print(self.s)

    def getPhi(self,x,c):
        if self.shape == 0:
            if abs(x - c) <= (1/self.m):
                return math.exp(-math.pow(x-c, 2))
            else:
                return 0
        elif self.shape == 1:
            return math.exp(-math.pow(x-c, 2)/(2*0.02**2))

    def getPhi_matrix(self):
        for i in range(self.lenth):
            self.Phi[i][i] = self.getPhi(self.x[i],self.ci)

    def getF(self):
        for i in range(self.lenth):
            self.f[i] = self.ddy0[i] - self.alpha*(self.beta*( self.y[self.lenth-1] - self.y[i]) - self.dy[i]) + self.alpha*self.beta*self.x[i]*( self.y[self.lenth-1] - self.y[0])

    def getW(self):
        self.getX()
        self.getS()
        self.getF()
        for i in range(self.m):
            self.ci = (i+1)/self.m
            self.getPhi_matrix()
            st_Phi = np.dot(self.s.T ,self.Phi )
            up = np.dot(st_Phi, self.f)
            down = np.dot(st_Phi, self.s)
            self.w[i] = up / down


    def getddy1(self):
        plot_y = np.zeros((self.lenth, 1), dtype=np.float)
        ddy1 = np.zeros((self.lenth, 1), dtype=np.float)
        dy1 = np.zeros((self.lenth, 1), dtype=np.float)
        self.getW()
        ddy1[0] = 0
        plot_y[0] = self.y[0]
        dy1[0] = self.dy[0]
        for i in range(1, self.lenth):
            up = 0
            down = 0
            for j in range(self.m):
                up += self.w[j]*self.getPhi(self.x[i],(j+1)/self.m)
                down += self.getPhi(self.x[i],(j+1)/self.m)
            #fx = (up/down)*self.x[i]*(self.y[self.lenth-1] - self.y[0])
            fx = (up/down)*self.x[i]*self.alpha*self.beta
            #ddy1[i] = (self.alpha*(self.beta* (self.y[self.lenth-1]+0.2*(self.y[self.lenth-1]-self.y[0]) - plot_y[i-1]) - dy1[i-1]) + fx)
            ddy1[i] = self.alpha*(self.beta* (self.y[self.lenth-1]+4.0*(self.y[self.lenth-1]-self.y[0]) - plot_y[i-1]) - dy1[i-1]) \
                     + fx - self.x[i]*self.alpha*self.beta* (self.y[self.lenth-1]+4.0*(self.y[self.lenth-1]-self.y[0]) - self.y[0])
            dy1[i] = dy1[i-1] + ddy1[i-1]*self.time_delta
            plot_y[i] = plot_y[i-1] + dy1[i-1]*self.time_delta
        return plot_y



class Bad_DynamicMP:
    alpha = 0.4
    beta = 0.8
    # self.m = 300
    freq = 120
    time_delta = 1 / freq
    ci = 0


    def __init__(self, m, y_all_i, dy_all_i, ddy0_all_i, lenth, shape=0, linear=0):

        # self.ci = i/8
        self.m = m
        self.shape = shape
        self.linear = linear
        self.y = y_all_i
        self.dy = dy_all_i
        self.ddy0 = ddy0_all_i
        self.lenth = lenth
        self.x = np.zeros((self.lenth, 1), dtype=np.float)
        self.s = np.zeros((self.lenth,1), dtype=np.float)
        self.Phi = np.zeros((self.lenth, self.lenth), dtype=np.float)
        self.f = np.zeros((self.lenth, 1), dtype=np.float)
        self.w = np.zeros((self.m, 1), dtype=np.float)


    def getX(self):
        for i in range(self.lenth):
            if i == 0:
                self.x[i] = 1
            else:
                if self.linear == 1:
                    self.x[i] = self.x[i-1] - self.time_delta*self.x[i-1]
                elif self.linear == 0:
                    self.x[i] = 1 - i/self.lenth

    def getS(self):
        tmp = (self.y[self.lenth-1] - self.y[0])
        for i in range(self.lenth):
            self.s[i] = self.x[i] * tmp
        # print(self.s)

    def getPhi(self,x,c):
        if self.shape == 0:
            if abs(x - c) <= (1/self.m):
                return math.exp(-math.pow(x-c, 2))
            else:
                return 0
        elif self.shape == 1:
            return math.exp(-math.pow(x-c, 2)/(2*0.02**2))

    def getPhi_matrix(self):
        for i in range(self.lenth):
            self.Phi[i][i] = self.getPhi(self.x[i],self.ci)

    def getF(self):
        for i in range(self.lenth):
            self.f[i] = self.ddy0[i] - self.alpha*(self.beta*( self.y[self.lenth-1] - self.y[i]) - self.dy[i])

    def getW(self):
        self.getX()
        self.getS()
        self.getF()
        for i in range(self.m):
            self.ci = (i+1)/self.m
            self.getPhi_matrix()
            st_Phi = np.dot(self.s.T ,self.Phi )
            up = np.dot(st_Phi, self.f)
            down = np.dot(st_Phi, self.s)
            self.w[i] = up / down


    def getddy1(self):
        plot_y = np.zeros((self.lenth, 1), dtype=np.float)
        ddy1 = np.zeros((self.lenth, 1), dtype=np.float)
        dy1 = np.zeros((self.lenth, 1), dtype=np.float)
        self.getW()
        ddy1[0] = 0
        plot_y[0] = self.y[0]
        dy1[0] = self.dy[0]
        for i in range(1, self.lenth):
            up = 0
            down = 0
            for j in range(self.m):
                up += self.w[j]*self.getPhi(self.x[i],(j+1)/self.m)
                down += self.getPhi(self.x[i],(j+1)/self.m)
            fx = (up/down)*self.x[i]*(self.y[self.lenth-1] - self.y[0])
            ddy1[i] = (self.alpha*(self.beta* (self.y[self.lenth-1]+0.2*(self.y[self.lenth-1]-self.y[0]) - plot_y[i-1]) - dy1[i-1]) + fx)
            dy1[i] = dy1[i-1] + ddy1[i-1]*self.time_delta
            plot_y[i] = plot_y[i-1] + dy1[i-1]*self.time_delta
        return plot_y