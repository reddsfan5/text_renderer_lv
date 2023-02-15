# -*- coding:utf-8 -*-
import base64
import colorsys
from pathlib import Path

from PIL import ImageFont, Image, ImageDraw
import numpy as np
import cv2
import random

import os
from fontTools.ttLib import TTFont, TTCollection
from tqdm import tqdm

from lmdbs.lmdbs.lmdb_saver import LmdbSaver


def chr_in_font(chrs, font, font_name):
    '''
    遍历chrs，确保每个字符都在uniMap中，且都在glyf表中（ttf）
    该方法只确保在规范字体中生效：
        如果，存在不规范字体，uniMap和glyf中都声明了某字符，但实际上该字符的字形用占位符糊弄，也会被认为是存在。
        如何判断某个汉字是不是在字体库中:https://cloud.tencent.com/developer/article/1576291
    '''
    try:
        # uniMap[23383]为uni5B57  ,hex(23383) 为 0x5b57.
        uniMap = font['cmap'].tables[0].ttFont.getBestCmap()  # uniMap是一个字典，字典的 key 是这个字体库中所有字符的 unicode 码
        # cff = font.get['CFF '] otf 字体没有glyf表，有个cff 表，但是字形映射在哪里记录着呢
    except (AssertionError, AttributeError) as e:
        print(f'{font["name"].names[3]}:{e}')
        return
    if not uniMap:
        print(f"Cmap为None字体：{font_name}")
        return None
    for char in chrs:
        if ord(char) not in uniMap:
            return False
    return True

def get_char():
    chars1 = '一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ①②③④⑤⑥⑦⑧⑨⑩1234567890'
    chars2 = r'啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂盎凹敖熬翱袄傲奥懊澳芭捌扒叭吧笆八疤巴拔跋靶把耙坝霸罢爸白柏百摆佰败拜稗斑班搬扳般颁板版扮拌伴瓣半办绊邦帮梆榜膀绑棒磅蚌镑傍谤苞胞包褒剥薄雹保堡饱宝抱报暴豹鲍爆杯碑悲卑北辈背贝钡倍狈备惫焙被奔苯本笨崩绷甭泵蹦迸逼鼻比鄙笔彼碧蓖蔽毕毙毖币庇痹闭敝弊必辟壁臂避陛鞭边编贬扁便变卞辨辩辫遍标彪膘表鳖憋别瘪彬斌濒滨宾摈兵冰柄丙秉饼炳病并玻菠播拨钵波博勃搏铂箔伯帛舶脖膊渤泊驳捕卜哺补埠不布步簿部怖擦猜裁材才财睬踩采彩菜蔡餐参蚕残惭惨灿苍舱仓沧藏操糙槽曹草厕策侧册测层蹭插叉茬茶查碴搽察岔差诧拆柴豺搀掺蝉馋谗缠铲产阐颤昌猖场尝常长偿肠厂敞畅唱倡超抄钞朝嘲潮巢吵炒车扯撤掣彻澈郴臣辰尘晨忱沉陈趁衬撑称城橙成呈乘程惩澄诚承逞骋秤吃痴持匙池迟弛驰耻齿侈尺赤翅斥炽充冲虫崇宠抽酬畴踌稠愁筹仇绸瞅丑臭初出橱厨躇锄雏滁除楚础储矗搐触处揣川穿椽传船喘串疮窗幢床闯创吹炊捶锤垂春椿醇唇淳纯蠢戳绰疵茨磁雌辞慈瓷词此刺赐次聪葱囱匆从丛凑粗醋簇促蹿篡窜摧崔催脆瘁粹淬翠村存寸磋撮搓措挫错搭达答瘩打大呆歹傣戴带殆代贷袋待逮怠耽担丹单郸掸胆旦氮但惮淡诞弹蛋当挡党荡档刀捣蹈倒岛祷导到稻悼道盗德得的蹬灯登等瞪凳邓堤低滴迪敌笛狄涤翟嫡抵底地蒂第帝弟递缔颠掂滇碘点典靛垫电佃甸店惦奠淀殿碉叼雕凋刁掉吊钓调跌爹碟蝶迭谍叠丁盯叮钉顶鼎锭定订丢东冬董懂动栋侗恫冻洞兜抖斗陡豆逗痘都督毒犊独读堵睹赌杜镀肚度渡妒端短锻段断缎堆兑队对墩吨蹲敦顿囤钝盾遁掇哆多夺垛躲朵跺舵剁惰堕蛾峨鹅俄额讹娥恶厄扼遏鄂饿恩而儿耳尔饵洱二贰发罚筏伐乏阀法珐藩帆番翻樊矾钒繁凡烦反返范贩犯饭泛坊芳方肪房防妨仿访纺放菲非啡飞肥匪诽吠肺废沸费芬酚吩氛分纷坟焚汾粉奋份忿愤粪丰封枫蜂峰锋风疯烽逢冯缝讽奉凤佛否夫敷肤孵扶拂辐幅氟符伏俘服浮涪福袱弗甫抚辅俯釜斧脯腑府腐赴副覆赋复傅付阜父腹负富讣附妇缚咐噶嘎该改概钙盖溉干甘杆柑竿肝赶感秆敢赣冈刚钢缸肛纲岗港杠篙皋高膏羔糕搞镐稿告哥歌搁戈鸽胳疙割革葛格蛤阁隔铬个各给根跟耕更庚羹埂耿梗工攻功恭龚供躬公宫弓巩汞拱贡共钩勾沟苟狗垢构购够辜菇咕箍估沽孤姑鼓古蛊骨谷股故顾固雇刮瓜剐寡挂褂乖拐怪棺关官冠观管馆罐惯灌贯光广逛瑰规圭硅归龟闺轨鬼诡癸桂柜跪贵刽辊滚棍锅郭国果裹过哈骸孩海氦亥害骇酣憨邯韩含涵寒函喊罕翰撼捍旱憾悍焊汗汉夯杭航壕嚎豪毫郝好耗号浩呵喝荷菏核禾和何合盒貉阂河涸赫褐鹤贺嘿黑痕很狠恨哼亨横衡恒轰哄烘虹鸿洪宏弘红喉侯猴吼厚候后呼乎忽瑚壶葫胡蝴狐糊湖弧虎唬护互沪户花哗华猾滑画划化话槐徊怀淮坏欢环桓还缓换患唤痪豢焕涣宦幻荒慌黄磺蝗簧皇凰惶煌晃幌恍谎灰挥辉徽恢蛔回毁悔慧卉惠晦贿秽会烩汇讳诲绘荤昏婚魂浑混豁活伙火获或惑霍货祸击圾基机畸稽积箕肌饥迹激讥鸡姬绩缉吉极棘辑籍集及急疾汲即嫉级挤几脊己蓟技冀季伎祭剂悸济寄寂计记既忌际妓继纪嘉枷夹佳家加荚颊贾甲钾假稼价架驾嫁歼监坚尖笺间煎兼肩艰奸缄茧检柬碱碱拣捡简俭剪减荐槛鉴践贱见键箭件健舰剑饯渐溅涧建僵姜将浆江疆蒋桨奖讲匠酱降蕉椒礁焦胶交郊浇骄娇嚼搅铰矫侥脚狡角饺缴绞剿教酵轿较叫窖揭接皆秸街阶截劫节桔杰捷睫竭洁结解姐戒藉芥界借介疥诫届巾筋斤金今津襟紧锦仅谨进靳晋禁近烬浸尽劲荆兢茎睛晶鲸京惊精粳经井警景颈静境敬镜径痉靖竟竞净炯窘揪究纠玖韭久灸九酒厩救旧臼舅咎就疚鞠拘狙疽居驹菊局咀矩举沮聚拒据巨具距踞锯俱句惧炬剧捐鹃娟倦眷卷绢撅攫抉掘倔爵觉决诀绝均菌钧军君峻俊竣浚郡骏喀咖卡咯开揩楷凯慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕颗科壳咳可渴克刻客课肯啃垦恳坑吭空恐孔控抠口扣寇枯哭窟苦酷库裤夸垮挎跨胯块筷侩快宽款匡筐狂框矿眶旷况亏盔岿窥葵奎魁傀馈愧溃坤昆捆困括扩廓阔垃拉喇蜡腊辣啦莱来赖蓝婪栏拦篮阑兰澜谰揽览懒缆烂滥琅榔狼廊郎朗浪捞劳牢老佬姥酪烙涝勒乐雷镭蕾磊累儡垒擂肋类泪棱楞冷厘梨犁黎篱狸离漓理李里鲤礼莉荔吏栗丽厉励砾历利僳例俐痢立粒沥隶力璃哩俩联莲连镰廉怜涟帘敛脸链恋炼练粮凉梁粱良两辆量晾亮谅撩聊僚疗燎寥辽潦了撂镣廖料列裂烈劣猎琳林磷霖临邻鳞淋凛赁吝拎玲菱零龄铃伶羚凌灵陵岭领另令溜琉榴硫馏留刘瘤流柳六龙聋咙笼窿隆垄拢陇楼娄搂篓漏陋芦卢颅庐炉掳卤虏鲁麓碌露路赂鹿潞禄录陆戮驴吕铝侣旅履屡缕虑氯律率滤绿峦挛孪滦卵乱掠略抡轮伦仑沦纶论萝螺罗逻锣箩骡裸落洛骆络妈麻玛码蚂马骂嘛吗埋买麦卖迈脉瞒馒蛮满蔓曼慢漫谩芒茫盲氓忙莽猫茅锚毛矛铆卯茂冒帽貌贸么玫枚梅酶霉煤没眉媒镁每美昧寐妹媚门闷们萌蒙檬盟锰猛梦孟眯醚靡糜迷谜弥米秘觅泌蜜密幂棉眠绵冕免勉娩缅面苗描瞄藐秒渺庙妙蔑灭民抿皿敏悯闽明螟鸣铭名命谬摸摹蘑模膜磨摩魔抹末莫墨默沫漠寞陌谋牟某拇牡亩姆母墓暮幕募慕木目睦牧穆拿哪呐钠那娜纳氖乃奶耐奈南男难囊挠脑恼闹淖呢馁内嫩能妮霓倪泥尼拟你匿腻逆溺蔫拈年碾撵捻念娘酿鸟尿捏聂孽啮镊镍涅您柠狞凝宁拧泞牛扭钮纽脓浓农弄奴努怒女暖虐疟挪懦糯诺哦欧鸥殴藕呕偶沤啪趴爬帕怕琶拍排牌徘湃派攀潘盘磐盼畔判叛乓庞旁耪胖抛咆刨炮袍跑泡呸胚培裴赔陪配佩沛喷盆砰抨烹澎彭蓬棚硼篷膨朋鹏捧碰坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏片骗飘漂瓢票撇瞥拼频贫品聘乒坪苹萍平凭瓶评屏坡泼颇婆破魄迫粕剖扑铺仆莆葡菩蒲埔朴圃普浦谱曝瀑期欺栖戚妻七凄漆柒沏其棋奇歧畦崎脐齐旗祈祁骑起岂乞企启契砌器气迄弃汽泣讫掐洽牵扦钎铅千迁签仟谦乾黔钱钳前潜遣浅谴堑嵌欠歉枪呛腔羌墙蔷强抢橇锹敲悄桥瞧乔侨巧鞘撬翘峭俏窍切茄且怯窃钦侵亲秦琴勤芹擒禽寝沁青轻氢倾卿清擎晴氰情顷请庆琼穷秋丘邱球求囚酋泅趋区蛆曲躯屈驱渠取娶龋趣去圈颧权醛泉全痊拳犬券劝缺炔瘸却鹊榷确雀裙群然燃冉染瓤壤攘嚷让饶扰绕惹热壬仁人忍韧任认刃妊纫扔仍日戎茸蓉荣融熔溶容绒冗揉柔肉茹蠕儒孺如辱乳汝入褥软阮蕊瑞锐闰润若弱撒洒萨腮鳃塞赛三叁伞散桑嗓丧搔骚扫嫂瑟色涩森僧莎砂杀刹沙纱傻啥煞筛晒珊苫杉山删煽衫闪陕擅赡膳善汕扇缮墒伤商赏晌上尚裳梢捎稍烧芍勺韶少哨邵绍奢赊蛇舌舍赦摄射慑涉社设砷申呻伸身深娠绅神沈审婶甚肾慎渗声生甥牲升绳省盛剩胜圣师失狮施湿诗尸虱十石拾时什食蚀实识史矢使屎驶始式示士世柿事拭誓逝势是嗜噬适仕侍释饰氏市恃室视试收手首守寿授售受瘦兽蔬枢梳殊抒输叔舒淑疏书赎孰熟薯暑曙署蜀黍鼠属术述树束戍竖墅庶数漱恕刷耍摔衰甩帅栓拴霜双爽谁水睡税吮瞬顺舜说硕朔烁斯撕嘶思私司丝死肆寺嗣四伺似饲巳松耸怂颂送宋讼诵搜艘擞嗽苏酥俗素速粟僳塑溯宿诉肃酸蒜算虽隋随绥髓碎岁穗遂隧祟孙损笋蓑梭唆缩琐索锁所塌他它她塔獭挞蹋踏胎苔抬台泰酞太态汰坍摊贪瘫滩坛檀痰潭谭谈坦毯袒碳探叹炭汤塘搪堂棠膛唐糖倘躺淌趟烫掏涛滔绦萄桃逃淘陶讨套特藤腾疼誊梯剔踢锑提题蹄啼体替嚏惕涕剃屉天添填田甜恬舔腆挑条迢眺跳贴铁帖厅听烃汀廷停亭庭艇通桐酮瞳同铜彤童桶捅筒统痛偷投头透凸秃突图徒途涂屠土吐兔湍团推颓腿蜕褪退吞屯臀拖托脱鸵陀驮驼椭妥拓唾挖哇蛙洼娃瓦袜歪外豌弯湾玩顽丸烷完碗挽晚皖惋宛婉万腕汪王亡枉网往旺望忘妄威巍微危韦违桅围唯惟为潍维苇萎委伟伪尾纬未蔚味畏胃喂魏位渭谓尉慰卫瘟温蚊文闻纹吻稳紊问嗡翁瓮挝蜗涡窝我斡卧握沃巫呜钨乌污诬屋无芜梧吾吴毋武五捂午舞伍侮坞戊雾晤物勿务悟误昔熙析西硒矽晰嘻吸锡牺稀息希悉膝夕惜熄烯溪汐犀檄袭席习媳喜铣洗系隙戏细瞎虾匣霞辖暇峡侠狭下厦夏吓掀锨先仙鲜纤咸贤衔舷闲涎弦嫌显险现献县腺馅羡宪陷限线相厢镶香箱襄湘乡翔祥详想响享项巷橡像向象萧硝霄削哮嚣销消宵淆晓小孝校肖啸笑效楔些歇蝎鞋协挟携邪斜胁谐写械卸蟹懈泄泻谢屑薪芯锌欣辛新忻心信衅星腥猩惺兴刑型形邢行醒幸杏性姓兄凶胸匈汹雄熊休修羞朽嗅锈秀袖绣墟戌需虚嘘须徐许蓄酗叙旭序畜恤絮婿绪续轩喧宣悬旋玄选癣眩绚靴薛学穴雪血勋熏循旬询寻驯巡殉汛训讯逊迅压押鸦鸭呀丫芽牙蚜崖衙涯雅哑亚讶焉咽阉烟淹盐严研蜒岩延言颜阎炎沿奄掩眼衍演艳堰燕厌砚雁唁彦焰宴谚验殃央鸯秧杨扬佯疡羊洋阳氧仰痒养样漾邀腰妖瑶摇尧遥窑谣姚咬舀药要耀椰噎耶爷野冶也页掖业叶曳腋夜液一壹医揖铱依伊衣颐夷遗移仪胰疑沂宜姨彝椅蚁倚已乙矣以艺抑易邑屹亿役臆逸肄疫亦裔意毅忆义益溢诣议谊译异翼翌绎茵荫因殷音阴姻吟银淫寅饮尹引隐印英樱婴鹰应缨莹萤营荧蝇迎赢盈影颖硬映哟拥佣臃痈庸雍踊蛹咏泳涌永恿勇用幽优悠忧尤由邮铀犹油游酉有友右佑釉诱又幼迂淤于盂榆虞愚舆余俞逾鱼愉渝渔隅予娱雨与屿禹宇语羽玉域芋郁吁遇喻峪御愈欲狱育誉浴寓裕预豫驭鸳渊冤元垣袁原援辕园员圆猿源缘远苑愿怨院曰约越跃钥岳粤月悦阅耘云郧匀陨允运蕴酝晕韵孕匝砸杂栽哉灾宰载再在咱攒暂赞赃脏葬遭糟凿藻枣早澡蚤躁噪造皂灶燥责择则泽贼怎增憎曾赠扎喳渣札轧铡闸眨栅榨咋乍炸诈摘斋宅窄债寨瞻毡詹粘沾盏斩辗崭展蘸栈占战站湛绽樟章彰漳张掌涨杖丈帐账仗胀瘴障招昭找沼赵照罩兆肇召遮折哲蛰辙者锗蔗这浙珍斟真甄砧臻贞针侦枕疹诊震振镇阵蒸挣睁征狰争怔整拯正政帧症郑证芝枝支吱蜘知肢脂汁之织职直植殖执值侄址指止趾只旨纸志挚掷至致置帜峙制智秩稚质炙痔滞治窒中盅忠钟衷终种肿重仲众舟周州洲诌粥轴肘帚咒皱宙昼骤珠株蛛朱猪诸诛逐竹烛煮拄瞩嘱主着柱助蛀贮铸筑住注祝驻抓爪拽专砖转撰赚篆桩庄装妆撞壮状椎锥追赘坠缀谆准捉拙卓桌琢茁酌啄着灼浊兹咨资姿滋淄孜紫仔籽滓子自渍字鬃棕踪宗综总纵邹走奏揍租足卒族祖诅阻组钻纂嘴醉最罪尊遵昨左佐柞做作坐座'
    chars = random.choice((chars1,chars2))
    char = random.choice(chars)
    return char


def get_color(bg_img):
    np_img = np.array(bg_img)
    mean = np.mean(np_img, axis=(0, 1))[:3]
    # 注意这里是RGB还是BGR。
    '''
    在HSV空间，1/2 是补色。
            values = [
        # rgb, hsv
        ((0.0, 0.0, 0.0), (  0  , 0.0, 0.0)), # black
        ((0.0, 0.0, 1.0), (4./6., 1.0, 1.0)), # blue
        ((0.0, 1.0, 0.0), (2./6., 1.0, 1.0)), # green
        ((0.0, 1.0, 1.0), (3./6., 1.0, 1.0)), # cyan
        ((1.0, 0.0, 0.0), (  0  , 1.0, 1.0)), # red
        ((1.0, 0.0, 1.0), (5./6., 1.0, 1.0)), # purple
        ((1.0, 1.0, 0.0), (1./6., 1.0, 1.0)), # yellow
        ((1.0, 1.0, 1.0), (  0  , 0.0, 1.0)), # white
        ((0.5, 0.5, 0.5), (  0  , 0.0, 0.5)), # grey
    ]
    '''
    color_h, color_s, color_v = colorsys.rgb_to_hsv(*(mean / 255).tolist())
    anti_h = color_h - random.uniform(.25, .5) if color_h > .5 else color_h + random.uniform(.25, .5)
    anti_s = random.uniform(.1, 1.0)
    anti_v = color_v - random.uniform(.3, .5) if color_v > .5 else color_v + random.uniform(.3, .5)

    anti_r, anti_g, anti_b = (np.array(colorsys.hsv_to_rgb(anti_h, anti_s, anti_v)) * 255).astype(np.uint8).tolist()

    alpha = np.random.randint(53,255)
    # todo lvxiaodong safe color   https://docs.python.org/zh-cn/3/library/colorsys.html
    text_color = (anti_r, anti_g, anti_b, alpha)
    return text_color

def grid_img(img_path,
             font_path=r'D:\lxd_code\OCR_SOURCE\font\font_set\简繁-简繁-低风险\常规类\975GothicSC-Bold.ttf',
             font_size=(30,42),
             density=.9,
             is_draw_box=False):
    '''
    一幅图一种字体，
    Parameters
    ----------
    img_path
    font_path
    font_size
    density
    chars

    Returns
    -------

    '''
    try:
        if font_path.endswith(("ttc", 'TTC')):
            ttc = TTCollection(font_path)
            # assume all ttfs in ttc file have same supported chars
            font_ft = ttc.fonts[0]

        elif (
                font_path.endswith("ttf")
                or font_path.endswith("TTF")
                or font_path.endswith("otf")
                or font_path.endswith("OTF")
        ):
            font_ft = TTFont(
                font_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1
            )
        else:
            return
    except:
        return

    print(img_path)
    img_arr = cv2.imread(img_path)
    if type(img_arr) is None:
        return
    img_arr = cv2.medianBlur(img_arr,ksize=5)
    ori_h,ori_w = img_arr.shape[:2]
    cut_ratio = .7
    start = random.uniform(0,1-cut_ratio-.1)
    start_x = int(start*ori_w)
    start_y = int(start*ori_h)
    cut_w = int(ori_w*cut_ratio)
    cut_h = int(ori_h*cut_ratio)
    img_arr = img_arr[start_y:start_y+cut_h,start_x:start_x+cut_w,:]
    img_arr = cv2.resize(img_arr,(1024,int(1024/img_arr.shape[1]*img_arr.shape[0])))
    img_h,img_w = img_arr.shape[:2]
    img_draw = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img_draw)
    font0 = ImageFont.truetype(font_path, font_size[1])
    x0, y0, x1, y1 = font0.getbbox('网')
    base_size = max(y1 - y0, x1 - x0)
    big_grid_size = base_size *8
    padding = base_size * 2
    gys = 0
    boxes = []
    while gys+big_grid_size<=img_h:
        gxs = 0
        while gxs+big_grid_size<=img_w:
            if random.randint(1,99)<int(100*density):
                xs = random.randint(int(gxs+padding),int(gxs+big_grid_size-padding))
                ys = random.randint(int(gys+padding),int(gys+big_grid_size-padding))
                cur_font_size = random.randint(*font_size)
                font1 = ImageFont.truetype(font_path, cur_font_size)
                char1 = get_char()
                if not chr_in_font(char1,font_ft,os.path.basename(font_path)):
                    # print(f'超纲字：{char1}')
                    continue


                x0, y0, x1, y1 = font1.getbbox(char1)
                # bbox = [[xs + x0, ys + y0], [xs + x1, ys + y1]]
                xs2 = xs+x0
                xe2 = xs+x1
                ys2 = ys+y0
                ye2 = ys+y1
                bbox = [[xs2,ys2],[xe2,ys2],[xe2,ye2],[xs2,ye2]]
                img_patch = img_arr[ys+y0:ys+y1,xs+x0:xs+x1,:]
                color = get_color(img_patch)


                draw.text((xs, ys), char1, color, font=font1)

                # 详细检测，识别格式，见：D:\lxd_code\callnumber\CallNumGenerator\genTools\commonProcess.py ->save_info_process
                sample = {
                             'transcription': char1,
                             'legibility': 1,
                             'points':bbox,
                             'label': ''
                }
                boxes.append(sample)
                if is_draw_box:
                    draw.rectangle(bbox, fill=None, outline='red')
            gxs += big_grid_size

        gys += big_grid_size
    img_draw = np.array(img_draw,np.uint8)
    save_info_db = {
        'image': img_draw,
        # 'img_size': bimg_aff_info['img_size'],
        'label': boxes,
        'file':os.path.basename(img_path)
    }

    return img_draw,save_info_db
if __name__ == '__main__':
    max_num = 20000
    lmdb_path = fr'D:\dataset\OCR\single_lmdb_num_{max_num}'
    lmdb_saver = LmdbSaver({'lmdb_path': lmdb_path, 'cnt': 0, 'cache_capacity': 500})
    img_dir = r'D:\dataset\ILSVRC2012mini\ILSVRC2012mini\val\part1'


    font_dir = r'D:\lxd_code\OCR_SOURCE\font\font_set\简繁-简繁-低风险\常规类'
    # 这里只使用ttf font
    font_paths = [str(f) for f in Path(font_dir).glob('**/*') if str(f).endswith('.ttf')]
    img_dir_path = Path(img_dir)
    count = 0
    img_paths = [str(img_path) for img_path in img_dir_path.glob('**/*') if str(img_path).endswith(('.jpg','.JPEG'))]
    for _ in tqdm(range(max_num)):
        # if count > max_num:
        #     break
        img_path = random.choice(img_paths)
        try:
            img_info = grid_img(str(img_path),font_path=random.choice(font_paths),is_draw_box=False)
        except:
            continue
        # count+=1
        if img_info:
            lmdb_saver.add(img_info[1],is_to_json=True)

        # s = base64.b64encode(os.urandom(3)).decode("utf8")
        # s = s.replace("\\", "").replace("/", "").replace("=","").replace("+","")
        # cv2.imwrite(os.path.join(r'D:\dataset\OCR\single_lmdb\samples',s+'.jpg'),im)
        # import matplotlib.pyplot as plt
        # plt.imshow(im)
        # plt.show()
    lmdb_saver.close()





