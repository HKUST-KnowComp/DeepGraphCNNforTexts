f = open("rcv1.topics.hier.orig.txt",'r')
lines = f.readlines()
nodes = []
for line in lines:
    keys = line.split(' ')
    while '' in keys:
        keys.remove("")
    node ={}
    node['parent'] = keys[1]
    node['child'] =keys[3]
    nodes.append(node)

f.close()

relation = {}
for node in nodes:
    parent = node['parent']
    child = node['child']
    if parent not in relation:
        relation[parent] = []
    relation[parent].append(child)

    
import json
result = []
with open('classes.json','r') as f:
    classes = json.load(f)
    for key in relation:
        if len(relation[key]) <2:
            continue
        new = []
        for index,values in enumerate(relation[key]):
            new.append(classes[values])
        result.append(new)
        
final = []      
for single in result:
    length = len(single)
    for i in range(length-1):
        for j in range(i+1,length):
            temp = []
            temp.append(single[i])
            temp.append(single[j])
            final.append(temp)
for v  in final:
    print(str(v))
with open ('heiring.json','w') as f:
    j = json.dump(final,f)
#print(j)
        
    
    