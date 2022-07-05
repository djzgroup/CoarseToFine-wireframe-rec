from sklearn.neighbors import KDTree
import numpy as np

def getAngle(x, y):
    l_x = np.sqrt(x.dot(x))
    l_y = np.sqrt(y.dot(y))
    dian = x.dot(y)
    cos_ = dian / (l_x * l_y)
    eps = 1e-6
    if 1.0 < cos_ < 1.0 + eps:
        cos_ = 1.0
    elif -1.0 - eps < cos_ < -1.0:
        cos_ = -1.0
    angle_hu = np.arccos(cos_)
    angle_d = angle_hu * 180 / np.pi
    return angle_d

def chamfer_distance(array1,array2):
    batch_size, num_point = array1.shape[:2]
    tree2 = KDTree(array2, leaf_size=num_point + 1)
    dist = np.zeros(batch_size,)
    for i in range(batch_size):
        tree1 = KDTree(array1[i], leaf_size=num_point+1)
        distances1, _ = tree1.query(array2)
        distances2, _ = tree2.query(array1[i])
        av_dist1 = np.mean(distances1)
        av_dist2 = np.mean(distances2)
        dist[i] = av_dist1 + av_dist2
    return dist

def geometry(y, ps1, ps2, cs):
    n = len(ps1) + len(ps2) + len(cs)
    vecs = []
    for i in range(y.shape[0]):
        vec = []
        for j in range(y.shape[1]//2):
            vec.append([y[i][j * 2] - y[i][j * 2 + 1]])
        vecs.append(vec)
    vecs = np.array(vecs).squeeze() #[200,24,3]
    loss = np.zeros(y.shape[0],)
    for i in range(vecs.shape[0]):
        a = 0
        for p,q in ps1:
            x1 = vecs[i][int(p)]
            y1 = vecs[i][int(q)]
            angle_d = getAngle(x1,y1)
            a = a + angle_d
        for p,q in ps2:
            x1 = vecs[i][int(p)]
            y1 = vecs[i][int(q)]
            angle_d = getAngle(x1,y1)
            a = a + 180 - angle_d
        for p,q in cs:
            x1 = vecs[i][int(p)]
            y1 = vecs[i][int(q)]
            angle_d = getAngle(x1, y1)
            a = a + abs(angle_d-90)
        if n > 0:
            loss[i] = a/n
    return loss

class PSO(object):
    def __init__(self, x, same, lines, ml, args):
        self.wmax = 0.9  # 惯性权重
        self.wmin = 0.4
        self.c1 = self.c2 = 2
        self.a = 1
        self.b = 0.01
        r = ml
        self.org = x #[48,3]
        self.vecs = []
        self.num = args.num_p
        for i in range(np.array(x).shape[0] // 2):
            self.vecs.append([x[i * 2] - x[i * 2 + 1]])
        self.vecs = np.array(self.vecs).squeeze() # [24,3]
        self.ps1 = []
        self.ps2 = []
        self.cs = []
        for i in range(self.vecs.shape[0]):
            x1 = self.vecs[i]
            for j in range(i + 1, self.vecs.shape[0]):
                y1 = self.vecs[j]
                angle_d = getAngle(x1, y1)
                if angle_d < 10:
                    self.ps1.append([i, j])
                if angle_d > 170 and angle_d < 180:
                    self.ps2.append([i, j])
                if angle_d > 80 and angle_d < 100:
                    self.cs.append([i, j])
        self.max_steps = args.pso_ite_step  # 迭代次数
        self.x = np.tile(x, (self.num,1,1))#[100,48,3]
        self.rd = np.zeros((self.num, len(same), 3)) # 初始化粒子群位置[100,16,3]
        self.same = same
        self.lines = lines
        self.v = np.random.uniform(-r, r, self.rd.shape)  # 初始化粒子群速度
        p = x
        line_density = 100
        self.scale = np.linspace(0, 1, line_density)
        samples = []
        for i in range(self.x.shape[1] // 2):
            for j in self.scale:
                samples.append(p[i * 2] + j * (p[i * 2 + 1] - p[i * 2]))
        fitness = self.calculate_fitness(np.tile(samples, (self.num,1,1)), self.x) #[100]
        self.p = self.rd  # 个体的最佳位置[100,16,3]
        self.pg = self.rd[np.argmin(fitness)]  # 全局最佳位置[16，3]
        self.individual_best_fitness = fitness  # 个体的最优适应度[100]
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度scalar

    def calculate_fitness(self, x, y):
        if self.a == 0:
            return self.b * geometry(y, self.ps1, self.ps2, self.cs)
        if self.b == 0:
            return self.a * chamfer_distance(x, self.lines)
        return self.a * chamfer_distance(x, self.lines) + self.b * geometry(y, self.ps1, self.ps2, self.cs)

    def evolve(self):
        for step in range(self.max_steps):
            self.w = self.wmax - (self.wmax - self.wmin) * (step / self.max_steps)
            xx = self.x
            r1 = np.random.rand(self.num, len(self.same), 3)
            r2 = np.random.rand(self.num, len(self.same), 3)
            # r1 = np.random.uniform(0, 1, (num, len(self.same), 3))
            # r2 = np.random.uniform(0, 1, (num, len(self.same), 3))
            # 更新速度和权重
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.rd) + self.c2 * r2 * (self.pg - self.rd)
            self.rd = self.v + self.rd
            for i in range(self.num):
                for j in range(len(self.same)):
                    xx[i][self.same[j]] = xx[i][self.same[j]] + self.rd[i][j]
            practical = []
            for k in range(self.num):
                p = xx[k]
                samples = []
                for i in range(self.x.shape[1] // 2):
                    for j in self.scale:
                        samples.append(p[i * 2] + j * (p[i * 2 + 1] - p[i * 2]))
                practical.append(samples)
            fitness = self.calculate_fitness(np.array(practical), xx)  # [100]
            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.rd[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.rd[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            print('%d, best fitness: %.5f, mean fitness: %.5f' % (step, self.global_best_fitness, np.mean(fitness)))
        p = np.array(self.org)
        for i in range(len(self.same)):
            p[self.same[i]] = p[self.same[i]] + self.pg[i]
        return p