import numpy as np


class Integral(object):

    def __init__(self, a, b, f, tol_err, tol_coe=10, max_recur=20):
        # \int_{a}^{b} f dx
        # tol_err: 误差限
        # tol_coe: 龙贝格算法加速一次后提高的精度\
        #   (理论值为15, 值越小算法的精度估计越保守)
        # max_recur: 自适应算法的二分区间次数的最大限制\
        #   (程序中表现为__adapt_inte_recur的最大递归层数)
        self.a = a
        self.b = b
        self.f = f
        self.tol_err = tol_err
        self.tol_coe = tol_coe
        self.max_recur = max_recur

    def simp_inte(self, a_s, b_s):
        # 对积分区间[a_s, b_s]使用辛普森公式
        # 返回S[a_s, b_x], S表示辛普森公式得到的近似积分值
        f_s = self.f
        s = (b_s - a_s) / 6 * (f_s(a_s) + 4 * f_s(0.5 * (a_s + b_s)) + f_s(b_s))
        return s

    def double_simp(self, a, b):
        # 返回s1 = S[a, b], s2 = S[a, c] + S[c, b], 其中c为a, b中点
        c = 0.5 * (a + b)
        s1 = self.simp_inte(a, b)
        s2 = self.simp_inte(a, c) + self.simp_inte(c, b)
        return s1, s2

    def __adapt_inte_recur(self, a, b, tol_remain, num_recur):
        # 该类的私有方法, 只用于被adapt_inite方法调用
        # 返回自适应的辛普森公式S_adapt[a, b]
        # tol_remain: 当前递归调用时允许的误差限
        # num_recur: 标记递归层数
        s1, s2 = self.double_simp(a, b)

        # 若此次递归达到精度要求, 则用龙贝格算法返回积分近似值
        if np.abs(s1 - s2) / self.tol_coe <= tol_remain:
            return 1 / 15 * (16 * s2 - s1)
        # 若此次递归达到了最大递归层数, 则说明在限定的递归次数下无法达到精度要求，并触发错误
        if num_recur is self.max_recur:
            raise RuntimeError("Unable to achieve accuracy, num_recur = %r." % num_recur)

        c = 0.5 * (a + b)
        s_l = self.__adapt_inte_recur(a, c, tol_remain / 2, num_recur + 1)
        s_r = self.__adapt_inte_recur(c, b, tol_remain / 2, num_recur + 1)
        return s_l + s_r

    def adapt_inite(self):
        # 返回辛普森自适应积分值
        return self.__adapt_inte_recur(self.a, self.b, self.tol_err, 0)


if __name__ == '__main__':

    def f_(x): return 1 / x ** 2


    inte_ = Integral(0.2, 1, f_, 10**-10)
    i_simp = inte_.simp_inte(0.2, 1) # 辛普森公式
    i_adapt = inte_.adapt_inite() # 自适应辛普森公式

    print('Integral by Simpson method = %r' % i_simp)
    print('Integral by adaptive Simpson method = %r' % i_adapt)
