import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules 


data = pd.read_excel('Online_Retail_Store.xlsx') 
data.head() 
data.info()
data.columns 
data.Country.unique() 

#Cleaning the Data-----------------
data.isnull().any()
data.isnull().sum()
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True) 
data.isnull().sum()

# Dropping all transactions which were done on credit ('C')
data.info() 
data['InvoiceNo'] = data['InvoiceNo'].astype('str') 
data = data[~data['InvoiceNo'].str.contains('C')] 

# Stripping extra spaces in the description 
data['Description'] = data['Description'].str.strip() 

#Splitting the data according to the region of transaction-------
# Transactions done in France 
basket_France = (data[data['Country'] =="France"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Transactions done in the United Kingdom 
basket_UK = (data[data['Country'] =="United Kingdom"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Transactions done in Portugal 
basket_Por = (data[data['Country'] =="Portugal"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

basket_Sweden = (data[data['Country'] =="Sweden"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

#Hot encoding the Data------------
def hot_encode(x): 
	if(x<= 0): 
		return 0
	if(x>= 1): 
		return 1

# Encoding the datasets 
basket_encoded = basket_France.applymap(hot_encode) 
basket_France = basket_encoded 

basket_encoded = basket_UK.applymap(hot_encode) 
basket_UK = basket_encoded 

basket_encoded = basket_Por.applymap(hot_encode) 
basket_Por = basket_encoded 

basket_encoded = basket_Sweden.applymap(hot_encode) 
basket_Sweden = basket_encoded 

#Building the models and analyzing the results-----------------

#France:
# Building the model 
frq_items = apriori(basket_France, min_support = 0.1, use_colnames = True) 
frq_items

# Collecting the inferred rules in a dataframe 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
print(rules.head()) 
France_rules=pd.DataFrame(rules)

#Portugal
frq_items = apriori(basket_Por, min_support = 0.15, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
print(rules.head()) 
Portugal_rules=pd.DataFrame(rules)

#Sweden
frq_items = apriori(basket_Sweden, min_support = 0.10, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
print(rules.head()) 
Sweden_rules=pd.DataFrame(rules)

#UK
frq_items = apriori(basket_UK, min_support = 0.05, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
print(rules.head()) 
UK_rules=pd.DataFrame(rules)


 
def draw_graph(rules, rules_to_show):
  import matplotlib.pyplot as plt
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16']   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in rules.iloc[i]['antecedents']:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i]['consequents']:
             
            G1.add_nodes_from([c])
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)
  plt.show()
 
 
#Chart showing association rules of few products frequently sold in France    
fig_1 = draw_graph (France_rules, 6)
fig_1

#Chart showing association of few products frequently sold in Portugal    
fig_2 = draw_graph (Portugal_rules, 4)
fig_2

#Chart showing association of few products frequently sold in Sweden    
fig_3 = draw_graph (Sweden_rules, 5)
fig_3

#Chart showing association of few products frequently sold in UK 
fig_4 = draw_graph (UK_rules, 5)
fig_4