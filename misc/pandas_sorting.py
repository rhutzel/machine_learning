import itertools
import pandas

index = list(itertools.product(['Ada','Quinn','Violet'],['Comp','Math','Sci']))
headr = list(itertools.product(['Exams','Labs'],['I','II']))
indx = pandas.MultiIndex.from_tuples(index,names=['Student','Course'])
cols = pandas.MultiIndex.from_tuples(headr)
data = [[70+x+y+(x*y)%3 for x in range(4)] for y in range(9)]

frame = pandas.DataFrame(data,indx,cols)
sorted_frame = frame.sort_values(by=('Labs', 'I'), ascending=False)
print(sorted_frame)
