import numpy as np

def random_selection(proto_number, train_number):
    # gets random prototypes
    return np.random.randint(train_number, size=proto_number)

def center_selection(proto_number, distances):
    # gets the center prototypes
    return np.argsort(np.sum(distances, axis=1))[:proto_number]

def border_selection(proto_number, distances):
    # gets the border prototypes
    return np.argsort(np.sum(distances, axis=1))[::-1][:proto_number]

def spanning_selection(proto_number, distances):
    # gets the spanning prototypes
    proto_loc = center_selection(1, distances)
    choice_loc = np.delete(np.arange(np.shape(distances)[0]), proto_loc, 0)
    for iter in range(proto_number-1):
        d = distances[choice_loc]
        p = np.array([choice_loc[np.argmax(np.min(d[:,proto_loc], axis=1))]])
        proto_loc = np.append(proto_loc, p)
        choice_loc = np.delete(choice_loc, p, 0)
    return proto_loc

def k_centers_selection(proto_number, distances):
    # finds k centers (k mediods)
    no_possible = np.shape(distances)[0]

    # initialize with spanning
    proto_loc = spanning_selection(proto_number, distances)
    for iter in range(1000):
        # assign every point into a group with the centers
        membership = np.zeros(no_possible, dtype=np.int32)
        for i, d in enumerate(distances):
            membership[i] = proto_loc[np.argmin(d[proto_loc])]

        # find center of groups
        was_change = False
        for i, p in enumerate(proto_loc):
            p_group = np.where(membership==p)[0]
            d_matrix = distances[p_group]
            new_center = p_group[center_selection(1, d_matrix[:,p_group])][0]
            if new_center != p:
                proto_loc[i] = new_center
                was_change = True
        if was_change == False:
            print("stopping at {}".format(iter))
            break
    return proto_loc


def selector_selector(selection, proto_number, distances):
    if selection == "random":
        return random_selection(proto_number, distances)
    elif selection == "centers":
        return center_selection(proto_number, distances)
    elif selection == "borders":
        return border_selection(proto_number, distances)
    elif selection == "spanning":
        return spanning_selection(proto_number, distances)
    elif selection == "kcenters":
        return k_centers_selection(proto_number, distances)
    else:
        exit("missing selection type")