import math
from pprint import pprint
class cm(object):
    pass


def angle_calculate(b: cm, c: cm, alpha, n: int, angle_class='angle'):
    '''
    对书架两个端点BC，连成的线段，均分为n份后，从C到B扫描，计算扫描角度，返回角度列表。
    假定：摄头到端点C的距离b 与摄头到端点B的距离c已知（可通过w,s,h等数据求得）。 b，c夹角 alpha已知。

    Parameters
    ----------
    b : 摄头到端点C的距离（∠B的对边）
    c ：摄头到端点B的距离（∠C 的对边）
    alpha ：∠A :默认角度制
    angle_class: 'angle' or 'arc'
    n ：均分数

    Returns：角度列表
    -------
    '''

    alpha = alpha * math.pi / 180 if angle_class == 'angle' else alpha

    a = math.sqrt(b ** 2 + c ** 2 - 2 * b * c * math.cos(alpha))
    e = a / n

    cos_beta = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    angel_last = 0
    angel_list = []
    for index in range(n):
        cur_e = (index + 1) * e

        # i1 摄像头 到 打点的线段长度
        i1 = math.sqrt(cur_e ** 2 + b ** 2 - 2 * cur_e * b * cos_beta)

        cos_alpha_multi = (i1 ** 2 + b ** 2 - cur_e ** 2) / (2 * i1 * b)
        alpha_multi = math.acos(cos_alpha_multi)  # 弧度：alpha_multi∠pAC
        alpha_multi_angel = alpha_multi * 180 / math.pi
        alpha_cur = alpha_multi_angel - angel_last
        angel_list.append(alpha_cur)
        angel_last = alpha_multi_angel

    return angel_list


if __name__ == '__main__':
    pprint(angle_calculate(321,336,1.41,5,angle_class='arc'))
