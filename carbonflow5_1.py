import numpy as np
import scipy.io as sio
import math
import numpy as np
import scipy.io as sio
import pandas as pd
def Btadjust(BTA, SOCA):
    if BTA >= 0:
        BTAA = min(BTA * 100, ((300 - SOCA)))
    else:
        BTAA = max(BTA * 100, ((30 - SOCA)))
    return BTAA

def updateSOC(SOCA, BTAA):
    SOCA = SOCA + BTAA
    return SOCA

class P2P:
    def __init__(self,num_agents=99):
        # 电池电量
        self.num_agents = num_agents
        self.SOCs = [50] * self.num_agents  # 用列表初始化100个智能体的SOC

        self.t = 0

    def reset(self):
        # 电池电量
        self.SOCs = [50] * self.num_agents  # 重置所有SOC为50
        self.t = 0
        # 初始化每个智能体的观察数据
        obs = [np.ones(6) for _ in range(self.num_agents)]  # 每个智能体的观察数据
        return obs


    def step(self, *actions):

            num_agents = 99
            pv_df = pd.read_excel('RESFL.xlsx', header=None)  # 假设pv数据不包含列名，24小时光伏发电
            load_df = pd.read_excel('LoadFL.xlsx', header=None)  # 假设load数据不包含列名，24小时负荷
            eai_df = pd.read_excel('EAI.xlsx', header=None)
            eai = np.array(eai_df[0].values)
            load_df = load_df.apply(pd.to_numeric, errors='coerce')
            pv_df = pv_df.apply(pd.to_numeric, errors='coerce')

            # 初始化存储负荷和光伏数据的列表
            load_list = []
            pv_list = []

            # 遍历每个社区的数据
            for i in range(load_df.shape[0]):  # 对每个社区
                load_data = load_df.iloc[i].values  # 获取第i个社区的负荷数据
                pv_data = pv_df.iloc[i].values  # 获取第i个社区的光伏数据

                # 可以在这里根据具体情况进行负荷和光伏数据的缩放
                load_data_scaled = np.multiply(load_data, 1.2)  # 可根据需求调整
                pv_data_scaled = np.multiply(pv_data, 0.7)  # 可根据需求调整

                load_list.append(load_data_scaled)  # 将负荷数据添加到列表
                pv_list.append(pv_data_scaled)  # 将光伏数据添加到列表
            # 将列表转换为numpy数组，以便后续处理
            load_array = np.array(load_list)
            pv_array = np.array(pv_list)



            pri_ToU = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.18, 0.18, 0.18, 0.18, 0.08, 0.08, 0.08, 0.08])
            price_buy = pri_ToU[self.t]
            price_fit = np.array([0.02, 0.02,0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.02, 0.02, 0.02,0.02])

            BTs = []  # 存储所有BT调整后的值
            updated_SOCs = []  # 存储更新后的SOC
            load_results = []
            acload_results=[]
            DG_results = []
            bidcoefficient=[]
            # 处理每个智能体的动作
            for i, action in enumerate(actions):
                BTA = action[0]  # 获取第i个智能体的BTA
                pri = 0.5 * (action[1] + 1)  # 获取pri值
                acload = 0.5 * (action[2] + 1)  # 获取acload值
                acload_results.append(acload)
                DG=  0.5 * (action[3]+1)
                # 调整BT并更新SOC
                BT_adjusted = Btadjust(BTA, self.SOCs[i])  # 动态获取对应智能体的SOC
                updated_SOC = updateSOC(self.SOCs[i], BT_adjusted)  # 更新该智能体的SOC
                DG_adjusted = DG*200
                # 存储调整后的BT和更新后的SOC
                BTs.append(BT_adjusted)
                DG_results.append(DG_adjusted)
                updated_SOCs.append(updated_SOC)

                # 更新该智能体的SOC（实际改变状态）
                self.SOCs[i] = updated_SOC
                bidding_price = price_fit[self.t] + pri * (price_buy - price_fit[self.t])

            # 遍历所有智能体并计算对应的负载
            for i in range(num_agents):
                load = acload_results[i] * load_array[i][self.t] *0.1
                load_results.append(load)
            P=np.zeros(num_agents)
            for i in range(num_agents):
                # 对每个社区计算功率
                P[i] = load_array[i][self.t] - load_results[i] - pv_array[i][self.t] + BTs[i]-DG_results[i]  # 计算社区A的功率

            # 创建竞价数组 bid，包含每个社区的编号、功率和价格
            bid = []
            for i in range(num_agents):
                # 每个社区的竞价：编号、功率、竞价价格
                bid.append([i + 1, P[i], bidding_price])  # i+1 是社区编号

            # 将 bid 转换为 numpy 数组以便后续处理
            bid = np.array(bid)

            B_bid = np.zeros((num_agents,3))
            S_bid = np.zeros((num_agents,3))
            eB_bid = np.zeros((num_agents, 3))
            eS_bid = np.zeros((num_agents, 3))
            threshold=4
            kappa=2
            for i in range(num_agents):
                if bid[i, 1] > 0:
                    B_bid[i, :] = bid[i, :]
                    if eai[i]<threshold:
                            eB_bid[i, 2] = bid[i, :]* kappa*1 / math.log(1 + eai[i])
                elif bid[i, 1] < 0:
                    S_bid[i, :] = bid[i, :]
                    if eai[i] < threshold:
                        eS_bid[i, 2] = bid[i, :]  * math.log(1 + eai[i])/  kappa
            # B_order = eB_bid[np.argsort(eB_bid[:, 2])[::-1]]  # equity-aware
            # sorted_indices = np.argsort(eB_bid[:, 2])[::-1]
            # B_bid_sorted = B_bid[sorted_indices]
            B_order = B_bid[np.argsort(B_bid[:, 2])[::-1]]
            sorted_indices = np.argsort(B_bid[:, 2])[::-1]
            B_bid_sorted = B_bid[sorted_indices]
            S_order = S_bid[np.argsort(S_bid[:, 2])]

            P_da = np.zeros((num_agents*2,1))
            c_da = np.zeros((num_agents*2,1))
            record_buyer = np.zeros((num_agents*2,1))
            record_seller = np.zeros((num_agents*2,1))

            m = 0  # 记录匹配的订单数
            for i in range(num_agents):
                if S_order[i, 0] != 0:  # 卖单存在
                    j = 0
                    while j < num_agents:  # 遍历买单
                        if B_order[j, 0] != 0 and B_order[j, 2] >= S_order[i, 2]:  # 买单价格高于或等于卖单
                            if m >= num_agents*2:  # 防止索引越界
                                break
                            # 计算成交量
                            matched_quantity = min(B_order[j, 1], -S_order[i, 1])  # 成交数量
                            if matched_quantity > 0:
                                # 记录交易信息
                                P_da[m, 0] = matched_quantity
                                record_buyer[m, 0] = B_order[j, 0]
                                record_seller[m, 0] = S_order[i, 0]
                                c_da[m, 0] = 0.5 * (B_bid_sorted[j, 2] + S_order[i, 2])
                                # 更新买单和卖单的剩余数量
                                B_order[j, 1] -= matched_quantity
                                S_order[i, 1] += matched_quantity
                                # 增加匹配计数
                                m += 1
                            # 检查买单和卖单是否已完成匹配
                            if B_order[j, 1] == 0:  # 当前买单匹配完成
                                B_order[j, 0] = 0  # 标记为已处理
                            if S_order[i, 1] == 0:  # 当前卖单匹配完成
                                S_order[i, 0] = 0  # 标记为已处理
                                break  # 卖单匹配完成后，退出内层循环
                        j += 1  # 尝试下一个买单


            grid = np.zeros((num_agents,1))
            for i in range(num_agents):
                for j in range(num_agents):
                    if B_order[j, 0] == i + 1:
                        grid[i,0] = B_order[j, 1]
                        break
                    elif S_order[j, 0] == i + 1:
                        grid[i,0] = S_order[j, 1]
                        break
                    else:
                        grid[i,0] = 0



            obs=np.zeros((num_agents,6))
            obs_list = []
            for i in range(num_agents):
                obs[i] = np.concatenate([
                    np.array([load_array[i][self.t]]) / 100,  # 负载数据，转换为一维数组
                    np.array([pv_array[i][self.t]]) / 100,  # 光伏数据，转换为一维数组
                    np.array([self.SOCs[i]]) / 100,  # SOC数据，转换为一维数组
                    np.array([grid[i, 0]]) / 100,  # 电网数据，转换为一维数组
                    np.array([price_fit[self.t]]),  # 当前价格，已经是数组形式
                    np.array([pri_ToU[self.t]])  # 时段电价，已经是数组形式
                ])
                obs_list.append(obs[i])

            obs_array = np.array(obs_list)

            P2P_cost = np.zeros((num_agents,1))
            for i in range(num_agents):
                if i+1  in record_buyer:
                    idx = np.where(record_buyer == i+1 )[0]
                    P2P_cost[i,0] = np.sum(c_da[idx,0] * P_da[idx,0])
                elif i+1 in record_seller:
                    idx = np.where(record_seller == i+1)[0]
                    P2P_cost[i,0] = -np.sum(c_da[idx,0] * P_da[idx,0])
            carboncost= np.zeros(num_agents)
            emission = np.zeros(num_agents)
            free = np.zeros(num_agents)
            allowance_co_df =  pd.read_excel('allowance.xlsx', header=None)
            for i in range(num_agents):
                emission[i] = 0.4*max(grid[i,0], 0) +0.2*pv_array[i][self.t]+0.5*DG_results[i]
                free[i] = allowance_co_df.iloc[i, 0]*max(grid[i,0], 0)+ allowance_co_df.iloc[i, 1]*pv_array[i][self.t]+allowance_co_df.iloc[i, 1]*DG_results[i]
                carboncost[i]= 0.2*max((emission[i]-free[i]),0)

            cost = np.zeros(num_agents)
            for i in range(num_agents):
                cost[i] =  price_fit[self.t] * min(grid[i,0], 0) + price_buy * max(grid[i,0], 0) + 0.05*abs(load_array[i][self.t]) + 0.003*abs(load_array[i][self.t])**2 +  carboncost[i]

            indi_cost =np.zeros((num_agents,1))
            for i in range(num_agents):
                indi_cost[i,0] = cost[i] + P2P_cost[i,0]

            RewardA = -sum(cost) / 10000
            shareob = obs_array.reshape(1, -1)
            self.t += 1
            if self.t > 23:
                isdone = True
            else:
                isdone = False

            return obs_array,obs_array.reshape(1, -1),np.full(num_agents, RewardA),np.full(num_agents, isdone),{},0, record_buyer, record_seller,P_da,indi_cost


