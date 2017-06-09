# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 10:09:29 2017

@author: dell
"""

import sys
from optparse import OptionParser

#样本点类
class Point(object):
    def __init__(self, items, category):
        self.items = items 
        self.category = category 

#ruleitem类，每个ruleitem由<condset, y>表示，其中condset为item集合，y为标签   
class Ruleitem(object):
    def __init__(self, items, y, condsup, rulesup):
        self.condset = items
        self.category = y
        self.condsup = condsup
        self.rulesup = rulesup

#AARC分类器类，使用self.rules对测试样的类别进行预测
class Classifier(object):
    def __init__(self):
        self.rules = []
        self.default_class = 'N'
        self.default_confidence = 0

#active association rule classification
class AARC(object):
    def __init__(self):            
        self.classifier = Classifier() 
        self.point_set = []
        self.minSup = 0
        self.minConf = 0
        self.retrain_ration = 0.1

    #输出所有ruleitem 
    def show_ruleitems(self, ruleitems):
        for ri in ruleitems:
            print(ri.condset, ri.category, ri.condsup, ri.rulesup)
    
    #对多个具有相同condset的ruleitem进行择优选择，只保留一个condset对应的ruleitem
    #选择的策略是：首先选取具有最大condsup的ruleitem，
    #若还有多个，则选取具有最大confidence的ruleitem，
    #若还有多个，则直接选取第一个
    def get_best(self, l):
        #print len(l)
        max_sup_list = []
        max_sup = 0
        #找到最大的condsup
        for item in l:
            if max_sup < item.condsup:
                max_sup = item.condsup
        #找到最大的condsup所对应的item集合
        for item in l:
            if item.condsup == max_sup:
                max_sup_list.append(item)
        #若该集合只有一个元素，则直接返回该元素
        if len(max_sup_list) == 1:
            return max_sup_list[0]
  
        max_conf = 0
        #找到最大的confidence
        for item in l:
            if max_conf < item.rulesup * 1.0 / item.condsup:
                max_conf = item.rulesup * 1.0 / item.condsup
        #此处不关心是否存在多个具有最大confidence的ruleitem，但至少存在一个
        #因此，不管是否存在多个，直接返回第一个confidence等于最大confidence的ruleitem，
        #此既满足寻找最大confidence的元素的要求，又满足顺序返回多个元素中的第一个的要求
        for item in l:
            #print item.condset, item.category, item.condsup, item.rulesup
            if item.rulesup * 1.0 / item.condsup == max_conf:
                return item
 
    #对ruleitems集合进行去重
    #去重的原因是：可能存在具有相同condset，但category不相同的ruleitems
    #去重的策略见self.get_best()
    def remove_duplicate(self, ruleitems):
        #ri_flag_list标识数组，用于避免重复处理具有相同condset的元素
        ri_flag_list = [0] * len(ruleitems)
 
        #print ruleitems
        s = []
        for i in range(0, len(ruleitems)):
            #tmp_list用于装载具有相同condset的元素
            tmp_list = []
            #对于未处理的，首先要把自己加进待处理集合
            tmp_list.append(ruleitems[i])
            #若已处理，则跳过，去处理下一个
            if ri_flag_list[i] == 1:
                continue
            ri_flag_list[i] = 1

            for j in range(i + 1, len(ruleitems)):
                if ri_flag_list[j] == 1:
                    continue

                if set(ruleitems[i].condset) == set(ruleitems[j].condset):
                    #把具有相同condset的元素加进待处理集合
                    tmp_list.append(ruleitems[j])
                    ri_flag_list[j] = 1

            #对具有相同condset的元素进行去重，选取最优的一个
            s.append(self.get_best(tmp_list))
        #print "s: "
        #self.show_ruleitems(s)
        #print "end"
        return s
 
    #得到1-ruleitems集合
    def get_large_1_ruleitems(self, point_set, minSup):
        ruleitems = []
        #遍历point_set中的每一个点
        for p in point_set:
            #遍历每个点的item集合中的每个item
            for item in p.items:
                find = False
                tmp = 1

                #遍历ruleitems集合中每个ruleitem，不能提前break，否则tmp值不是最新的
                for ri in ruleitems:
                    #判断该ruleitem中的item集合(condset)是否和[item]相等
                    if set(ri.condset) == set([item]):
                        #若相等，则该ruleitem的condsup加一
                        ri.condsup += 1
                        tmp = ri.condsup
                        #判断分类是否相同，若相同，则该ruleitem的rulesup加一
                        if ri.category == p.category:
                            find = True
                            ri.rulesup += 1

                #如果当前ruleitems无此ruleitem，则添加新的ruleitem
                if find == False:
                    new_ri = Ruleitem([item], p.category, tmp, 1)
                    ruleitems.append(new_ri)

        #self.show_ruleitems(ruleitems)
        #精简ruleitems集合，挑出符合规定的ruleitem
        prime_ris = self.remove_duplicate(ruleitems)
        #print "prime_ris: "
        #self.show_ruleitems(prime_ris)
        #print "pend"
        large_set = []
        for ri in prime_ris:
            if ri.condsup * 1.0 / len(point_set) >= minSup: 
                large_set.append(ri)
         
        #print "large_set: "
        #self.show_ruleitems(large_set)
        #print "lend"
        return large_set
    
    #由ruleitems和设定的minConf得到rules集合
    def gen_rules(self, ruleitems, minConf):
        rules = []
        for ri in ruleitems:
            if ri.rulesup * 1.0 / ri.condsup >= minConf:
                rules.append(ri)
        return rules
    
    #由candidate_set和设定的minSup得到ruleitems集合
    def get_ruleitems(self, candidate_set, point_set, minSup):
        ruleitems = []
        for ri in candidate_set:
            if ri.condsup * 1.0 / len(point_set) >= minSup:
                ruleitems.append(ri)
        return ruleitems
    
    #得到一个集合的所有最大真子集
    def largest_true_subsets(self, s):
        l = list(s)
        for i in range(0, len(l)):
            s1 = l[0 : i] 
            s2 = l[i + 1: len(l)]
            yield s1 + s2

    #判断condset为sub的ruleitem是否在ruleitems集合中
    def is_in_ruleitems(self, ruleitems, sub):
        for ri in ruleitems:
            if set(ri.condset) == set(sub):
                return True
        return False

    #得到两个集合的交集
    def get_intersection(self, l1, l2):
        la = l1 
        lb = l2
        if len(l1) > len(l2):
            la = l2
            lb = l1
        intersection = []
        for item in la:
            if item in lb:
                intersection.append(item)

        return intersection

    #得到两个集合的并集
    def get_union(self, l1, l2):
        return l1 + l2

    #通过连接、剪枝的得到candidate_set集合
    def candidate_gen(self, old_ruleitems):
        new_ruleitems = []
        
        #self.show_ruleitems(items_set)
        #连接两个相同的ruleitems
        for ri in old_ruleitems:
            for rj in old_ruleitems:
                #print("isok?")
                #print(ri.condset, rj.condset)
                #如果ri和rj同属于一个类别，则他们之间的连接有效
                if ri.category == rj.category:
                    
                    intersection = self.get_intersection(ri.condset, rj.condset)
                    #print(intersection)
                    #如果他们之间不同的元素个数为1，则连接
                    if len(ri.condset) - len(intersection)  == 1:
                        #连接
                        union = self.get_union(ri.condset, rj.condset)
                        ok = True
                        #判断连接后的集合的子集是否还在原来的ruleitems中
                        for sub in self.largest_true_subsets(union):
                            if self.is_in_ruleitems(old_ruleitems, sub) == False:
                                ok = False
                                break
                        #如果在，则连接有效
                        if ok == True:
                            find = False
                            for nri in new_ruleitems:
                                if set(nri.condset) == set(union): #and nri.category == ri.category:
                                    find = True
                                    break
                            if find == False:
                                nri = Ruleitem(union, ri.category, 0, 0)
                                new_ruleitems.append(nri)
                                
        new_ruleitems = self.remove_duplicate(new_ruleitems)
        #self.show_ruleitems(new_ruleitems)
        return new_ruleitems

    #判断l1是否是l的子集
    def issubset(self, l1, l):
        for item in l1:
            if item not in l:
                return False
        return True

    #得到candidate_set集合中为point的子集的元素
    def rule_subset(self, candidate_set, point):
        sub_set = []
        for ruleitem in candidate_set:
            if self.issubset(ruleitem.condset, point.items) == True:
                sub_set.append(ruleitem)
        return sub_set
    
    #apriori算法，依照原论文实现
    def apriori(self, point_set, minSup, minConf):
        
        k = 1
        #得到1-ruleitems集合，其中每个元素的support大于等于minSup
        f1 = self.get_large_1_ruleitems(point_set, minSup)
        #从1-ruleitems中得到confidence大于等于minConf的元素
        cars = self.gen_rules(f1, minConf)
        #self.show_ruleitems(f1)
        current_ruleitems = f1
        #print(cars)
        while current_ruleitems != []:
            k += 1
            #由current_ruleitems集合通过连接、剪枝生成candidate_set集合
            #print "pre k: ", k
            ck = self.candidate_gen(current_ruleitems)
            #self.show_ruleitems(ck)
            #self.show_ruleitems(current_ruleitems)
            for p in point_set:
                #取出candidate_set集合中所有其condset为p.items的子集的ruleitem
                cd = self.rule_subset(ck, p)
                #print(p.items, p.category)
                #self.show_ruleitems(cd)
                #print(p.items, p.category)
                #self.show_ruleitems(cd)
                #对ruleitem的condsup加一
                for c in cd:
                    c.condsup += 1
                    #若类别也相同，则rulesup加一
                    if p.category == c.category:
                        c.rulesup += 1
            #print "post k: ", k
            #self.show_ruleitems(ck)
            #对candidate集合进行进一步处理，得到large_item,即k-ruleitems
            #print("end")
            current_ruleitems = self.get_ruleitems(ck, point_set, minSup)
            #self.show_ruleitems(current_ruleitems)
            #对上一步得到的ruleitems进行进一步处理，得到rules，即confidence大于都等于minConf的ruleitem
            tmp = self.gen_rules(current_ruleitems, minConf)
            cars = self.get_union(cars, tmp)
            #print("end")
        #self.show_ruleitems(cars)
        return cars

    #排序函数,排序策略为：先比较confidence，若相同则比较support，若相同则顺序选择第一个
    def myCmp(self, ia, ib):
        iaconf = ia.rulesup * 1.0 / ia.condsup
        ibconf = ib.rulesup * 1.0 / ib.condsup

        if iaconf > ibconf:
            return 1
        elif iaconf < ibconf:
            return -1
        elif ia.condsup >= ib.condsup:
            return 1
        else:
            return -1

    #对所有rule进行排序
    def sort_rules(self, rules):
        rules.sort(cmp = self.myCmp, reverse = True)
        return rules
        
    #得到数量最多的分类
    def get_max_count_class(self, point_set):
        class_dict = {}
        for i in range(0, len(point_set)):
            if point_set[i].category not in class_dict:
                class_dict[point_set[i].category] = 0
            class_dict[point_set[i].category] += 1

        max_count = 0
        for key in class_dict:
            if max_count < class_dict[key]: 
                max_count_class = key
                max_count = class_dict[key]

        return max_count_class, class_dict[max_count_class] * 1.0 / len(point_set)

    #得到当前默认的分类
    def get_current_default_class(self, point_set, point_flag_list):
        class_dict = {}
        ok = False
        default_class = point_set[0].category
        for i in range(0, len(point_set)):
            if point_flag_list[i] == 1:
                continue
            ok = True
            if point_set[i].category not in class_dict:
                class_dict[point_set[i].category] = 0
            class_dict[point_set[i].category] += 1

        max_count = 0
        for key in class_dict:
            if max_count < class_dict[key]: 
                default_class = key
                max_count = class_dict[key]

        if ok == False:
            return self.get_max_count_class(point_set), ok

        return default_class, class_dict[default_class] * 1.0 / len(point_set), ok

    #得到当前规则集的分类错误成本
    def get_current_cost(self, rules, rule_flag_list, default_class, point_set):
        cost = 0
        for point in point_set:
            ok = False
            for i in range(0, len(rule_flag_list)):
                if rule_flag_list[i] == 1:
                    if self.issubset(rules[i].condset, point.items): 
                        if rules[i].category == point.category:
                            ok = True
                            break
                        else:
                            cost += 1
            if ok == False and point.category != default_class:
                cost += 1
            
        return cost

    #计算成本链表
    def compute_cost_list(self, rules, point_set):
        point_flag_list = [0] * len(point_set)
        rule_flag_list = [0] * len(rules)
        cost_list = []
        
        for i in range(0, len(rules)):
            have = False
            for j in range(0, len(point_set)):
                if point_flag_list[j] == 1:
                    continue
                have = True
                if self.issubset(rules[i].condset, point_set[j].items) and \
                        rules[i].category == point_set[j].category:
                    rule_flag_list[i] = 1
                    point_flag_list[j] = 1
            if have == False:
                break
            default_class, default_confidence, ok = self.get_current_default_class(point_set, point_flag_list)
            cost_list.append([i, default_class, self.get_current_cost(rules, rule_flag_list, default_class, point_set)])
            if ok == False:
                break

        return cost_list, default_confidence

    #去掉给分类器带来更多误差的规则，剩下的则是最好的规则集
    def get_best_rules(self, cost_list, rules):
        min_cost = cost_list[0][2]
        p = 0
        best_rules = []
        for i in range(1, len(cost_list)):
            if cost_list[i][2] < min_cost:
                min_cost = cost_list[i][2]
                p = i
        for i in range(0, p + 1):
            best_rules.append(rules[i])

        return best_rules

    #用指定的参数训练样本，得到分类器
    def train(self, point_set, minSup, minConf):
        #self.point_set = self.get_training_point_set(fname)
        self.point_set = point_set
        self.minSup = minSup
        self.minConf = minConf
        
        #print(minSupport, minConfidence)
        rules = self.apriori(self.point_set, minSup, minConf)
        rules = self.sort_rules(rules)
        #self.show_ruleitems(rules)
        cost_list, self.classifier.default_confidence = self.compute_cost_list(rules, self.point_set)
        #print 'cost_list: ', cost_list
        self.classifier.rules = self.get_best_rules(cost_list, rules)
        self.classifier.default_class = cost_list[len(cost_list) - 1][1]
    
        self.show_ruleitems(self.classifier.rules)
        print "classifier default class: ", self.classifier.default_class
        #self.show_ruleitems(self.best_rules)
        
    #采用“积极策略”重新训练数据集
    #其中包括按顺序删除训练集中的一些旧样本点，以及从测试集中选取一些新样本点加入训练集
    def active_retrain(self, test_point_set, result_list, category_list):
        #对分类结果按照confidence值的大小排序
        result_list = sorted(result_list, key = lambda result: result[2], reverse = True)

        #取confidence值前(self.retrain_ratio)%大的测试样本点
        pre_index = int(len(test_point_set) * self.retrain_ration)
        #取confidence值后(self.retrain_ratio)%小的测试样本点
        post_index = len(test_point_set) - pre_index

        #按顺序移除之前的训练样本点
        reduced_count = 2 * pre_index
        if len(self.point_set) < reduced_count:
            if len(self.point_set) > pre_index:
                reduced_count /= 2
            else:
                reduced_count = 0
        for i in range(0, reduced_count):
            self.point_set.__delitem__(i)

        #添加新的样本点
        for i in range(0, pre_index):
            self.point_set.append(Point(test_point_set[result_list[i][0]].items, category_list[result_list[i][0]]))
        for i in range(post_index, len(test_point_set)):
            self.point_set.append(Point(test_point_set[result_list[i][0]].items, category_list[result_list[i][0]]))

        self.train(self.point_set, self.minSup, self.minConf)

    #对测试集样本点进行类别预测
    #返回列表类型的测试结果，其中每个元素的类型为：[测试样本点编号，测试类别，确信度]
    def classify(self, test_point_set):    
        #test_point_set = self.get_test_point_set(fname)
        result_list = []
        for i in range(0, len(test_point_set)):
            ok = False
            for rule in self.classifier.rules:
                if self.issubset(rule.condset, test_point_set[i].items): 
                    result = rule.category
                    confidence = rule.rulesup * 1.0 / rule.condsup
                    ok = True
                    break
            if ok == False:
                result = self.classifier.default_class
                confidence = self.classifier.default_confidence
            result_list.append([i, result, confidence])
        #self.active_rerain(result_list, y)
    
        return result_list

#从文件中得到测试数据集
def get_testing_dataset(fname):
    items_list = []
    file_iter = open(fname, 'r')
    for line in file_iter:
        # Remove trailing comma
        line = line.strip().rstrip(',')         
        # print("line2: ", line)
        #new_keyword
        record = line.split(',')
        # print("record: ", record)
        #new_keyword
        #print(record)
        #point_set.append(Point(record, 'N'))
        items_list.append(record)

    return items_list

#从文件中得到训练数据集
def get_training_dataset(fname):
    items_list = []
    category_list = []
    file_iter = open(fname, 'r')
    for line in file_iter:
        # Remove trailing comma
        line = line.strip().rstrip(',')         
        # print("line2: ", line)
        #new_keyword
        record = line.split(',')
        # print("record: ", record)
        #new_keyword
        #print(record)
        #point_set.append(Point(record[0 : len(record) - 1], record[len(record) - 1]))
        items_list.append(record[0 : len(record) - 1])
        category_list.append(record[len(record) - 1])

    return items_list, category_list

#生成训练样本点
def gen_training_point_set(items_list, category_list):
    point_set = []
    for i in range(0, len(items_list)):
        point_set.append(Point(items_list[i], category_list[i]))

    return point_set

#生成测试样本点
def gen_testing_point_set(items_list):
    point_set = []
    for i in range(0, len(items_list)):
        point_set.append(Point(items_list[i], 'N'))

    return point_set

#输出所有样本点的内容
def show_point_set(point_set):
    for point in point_set:
        print(point.items, point.category)

#得到测试集的分类正确率
def get_accuracy(point_set, result_list):
    right_count = 0
    for i in range(0, len(point_set)):
        if point_set[i].category == result_list[i][1]:
            right_count += 1

    return right_count * 1.0 / len(point_set)

#从样本点集中得到分类列表
def get_category_list_from_point_set(point_set):
    category_list = []
    for point in point_set:
        category_list.append(point.category)

    return category_list

def main():
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
                         default=None)
    optparser.add_option('-s', '--minSupport',
                         dest='minS',
                         help='minimum support value',
                         default=0.15,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='minC',
                         help='minimum confidence value',
                         default=0.6,
                         type='float')

    (options, args) = optparser.parse_args()

    if options.input is None:
        print('No dataset filename specified, system with exit\n')
        sys.exit('System will exit')

    #show_point_set(point_set)
    minSupport = options.minS
    minConfidence = options.minC
    
    training_items_list, training_category_list = get_training_dataset(options.input)
    training_point_set = gen_training_point_set(training_items_list, training_category_list)

    ratio = 0.9
    aarc = AARC()
    
#    print aarc.classifier
    i = 0
    sub_point_set = training_point_set[int(round(0.1 * i * len(training_point_set))) : int(round(0.1 * (i + 1) * len(training_point_set)))]
#    sub_training_point_set = sub_point_set[0, ratio * len(sub_point_set)]
#    sub_testing_point_set = sub_point_set[ration * len(sub_point_set), len(sub_point_set)]
    aarc.train(sub_point_set, minSupport, minConfidence)

    for i in range(1, 10):
        sub_point_set = training_point_set[int(round(0.1 * i * len(training_point_set))) : int(round(0.1 * (i + 1) * len(training_point_set)))]
        result_list = aarc.classify(sub_point_set)
        accuracy = get_accuracy(sub_point_set, result_list)
        print i, ": ", accuracy
        category_list = get_category_list_from_point_set(sub_point_set)
        aarc.active_retrain(sub_point_set, result_list, category_list)
    
if __name__ == "__main__":
    main()
