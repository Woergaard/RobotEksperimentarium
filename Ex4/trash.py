#Kopieret fra Steer funktionen.

    
#edge = make_edge(nearest_node, steering_node)
    #nævner = np.sqrt(edge.x**2 + edge.z**2)
    #x = nearest_node.x - (h.x*stepLength)/nævner
    #z = nearest_node.z - (edge.z*stepLength)/nævner

    #nævner = np.sqrt(steering_node.x**2 + steering_node.z**2)
    #x = (nearest_node.x - steering_node.x)  / linalg.norm(nearest_node.x - steering_node.x)#nævner + steering_node.x
    #z = (nearest_node.z - steering_node.z) / linalg.norm(nearest_node.z - steering_node.z) #nævner + steering_node.z
    
    #print("x:", x, "z:", z)

    #x = steering_node.x + x * stepLength
    #z = steering_node.z + z * stepLength

    #print("x:", x, "z:", z)

    ### Asgers Numpy Vectors ###
    #near_node = np.array([nearest_node.x, nearest_node.z])
    #steer_node = np.array([steering_node.x, steering_node.z])
    #xz = near_node - steer_node
    #magnitude = np.sqrt(xz[0]**2 + xz[1]**2) # vectors længde
    #new_vec = (xz * stepLength)/magnitude
    ### END ###