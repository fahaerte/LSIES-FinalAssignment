# useful pieces of code
monday = datetime.strptime('2023.05.01 00:00:00', '%Y.%m.%d %H:%M:%S')
friday = datetime.strptime('2023.05.05 23:59:59', '%Y.%m.%d %H:%M:%S')
saturday = datetime.strptime('2023.05.07 00:00:00', '%Y.%m.%d %H:%M:%S')
sunday = datetime.strptime('2023.05.07 23:59:59', '%Y.%m.%d %H:%M:%S')
woindex = pd.date_range(monday, friday, freq='1min')
weindex = pd.date_range(saturday, sunday, freq='1min')


def get_close_edges(df_to_add, nearest_edges, graph):
    i = 0
    mm = len(df_to_add)
    for edge1 in df_to_add['edges']:
        i = i+1
        if(i%10 < .2):
            print(i/mm*100)
            
            
        tname = df_to_add['name'][edge1]
        idx1 = (nearest_edges['name'] == tname)
        #idx2 = ((df_to_add['name'] == tname) * (df_to_add['edges'] != edge1))
        
        # exclude unwanted edges
        #tnot = df_to_add['nn_edge'][edge1]
        #pp = []
        #if len(tnot) > 0:
            #print(edge1)
            #print(tnot)
        #    pp = [(df_to_add['edges'] != ed) for ed in tnot];
        #    for p in pp:
        #        idx2 = idx2*p;
            
        #print(idx2.sum())
       # edges = pd.concat([nearest_edges[idx1]['edges'], df_to_add[idx2]['edges']]);
        edges = nearest_edges[idx1]['edges']
        #print(len(edges))
        #edges = edges.drop(edges == edge1)
        #print(len(edges))
        total_distance =[]

        for edge2 in edges:
            edge1_start = graph.nodes[edge1[0]]['y'], graph.nodes[edge1[0]]['x']
            edge1_end = graph.nodes[edge1[1]]['y'], graph.nodes[edge1[1]]['x']
            edge2_start = graph.nodes[edge2[0]]['y'], graph.nodes[edge2[0]]['x']
            edge2_end = graph.nodes[edge2[1]]['y'], graph.nodes[edge2[1]]['x']


            distance1 = geodesic(edge1_start, edge2_start).m
            distance2 = geodesic(edge1_end, edge2_end).m
            distance3 = geodesic(edge1_start, edge2_end).m
            distance4 = geodesic(edge1_end, edge2_start).m

            # Total distance between the edges
            total_distance.append(min([distance1, distance2, distance3, distance4]))
        
        if(min(total_distance) < 1000):    
            nearest = edges[total_distance.index(min(total_distance))]
            if nearest in nearest_edges.index:
                df_to_add['n_edge'][edge1] = nearest
            
            # see if two edges don't point on each other
            elif df_to_add['n_edge'][nearest] == df_to_add['edges'][edge1]:
                # save current nearest into the unwanted
                ii = total_distance.index(min(total_distance))
                df_to_add['nn_edge'][edge1].append(nearest)
                
                # find the 2nd nearest
                total_distance.pop(ii)
                nearest = edges[total_distance.index(min(total_distance))]
                df_to_add['n_edge'][edge1] = nearest
            else:
                df_to_add['n_edge'][edge1] = nearest
        else:
            df_to_add.drop(edge1);
        
        #del distance1, distance2, distance3, distance4, nearest, #total_distance, edge1, edge2, edges, edge1_end, edge1_start, #edge2_end, edge2_start
        #del tname, idx1, idx2
        
    return df_to_add



## fill in further
l1 = len(df_to_add)
l2 = 0
while l1-l2 > 0:
    l1 = df_to_add['ddB'].isna().sum()
    temp = df_to_add[df_to_add.isna()]
    for edge in temp.index:
        nedge = df_to_add['n_edge'][edge]
        if nedge in df_to_add['edges']:
            df_to_add['ddB'][edge] = df_to_add['ddB'][nedge]
            df_to_add['edB'][edge] = df_to_add['edB'][nedge]
            df_to_add['ndB'][edge] = df_to_add['ndB'][nedge]
    
    l2 = df_to_add['ddB'].isna().sum()
    
print(df_to_add['ddB'].isna().sum())

wrong concat - that why not working