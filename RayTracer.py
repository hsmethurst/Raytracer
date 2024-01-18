import sys
import numpy as np


path = str(sys.argv[1])

with open(path) as f:
    lines = f.readlines()
	
s_array = []
l_array = []    

for line in lines:
    line_header = str(line).split()[0]
    match line_header:
        case 'NEAR':
            near = float(line.split()[1])
        case 'LEFT':
            left = float(line.split()[1])
        case 'RIGHT':
            right = float(line.split()[1])
        case 'BOTTOM':
            bottom = float(line.split()[1])
        case 'TOP':
            top = float(line.split()[1])
        case 'RES':
            res = list(map(int, line.split()[1:]))
        case 'SPHERE':
            s_array.append(list(map(float, line.split()[2:])))
        case 'LIGHT':
            l_array.append(list(map(float, line.split()[2:])))
        case 'BACK':
            back = list(map(float, line.split()[1:]))
        case 'AMBIENT':
            ambient = list(map(float, line.split()[1:]))
        case 'OUTPUT':
            output = line.split()[1]

ray_origin = np.array([0.0,0.0,0.0,1]).T

# was having problems with np.linalg.norm() and so I made this to get magnitude of vectors
def get_magnitude(vector):
    return np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    
def calculate_ambient(intersected_obj):
    object_colors = intersected_obj[6:9]
    ka = intersected_obj[9]
    colors = np.array([0.0, 0.0, 0.0])
    for c in range(0, 3):
        ambient_color = ambient[c]
        object_color = object_colors[c]
        color = ka*ambient_color*object_color
        colors[c] = color
    return colors
    
def calculate_illumination(intersect, normal, t_prime, intersected_obj, ray):
    ray_copy = ray / get_magnitude(ray)
    sum = np.array([0.0, 0.0, 0.0])
    for light in l_array:
        shadow = False
        for s in s_array:
            if is_shadow(intersect, light, s, normal):
                # there is a shadow
                shadow = True
                break
        if not shadow:
            colors = np.array([0.0, 0.0, 0.0])
            light_colors = light[3:6]
            object_colors = intersected_obj[6:9]
            kd = intersected_obj[10]
            ks = intersected_obj[11]
            shininess = intersected_obj[13]
            light_pos = light[0:3]
            light_pos.append(1)
            L = np.array(light_pos) - np.array(intersect)
            L = (L[0:3] / get_magnitude(L)).tolist()
            L.append(0)
            L = np.array(L)
            R = 2 * np.dot(normal, (-L).T) * normal - L
            R = (R[0:3] / get_magnitude(R.tolist()[0])).tolist()
            R = np.array(R)
            for c in range(0, 3):
                light_color = light_colors[c]
                object_color = object_colors[c]
                
                color = kd*light_color*np.abs(np.dot(normal, (-L).T))*object_color + ks*light_color*np.power((np.dot(R, -ray_copy)), shininess)
                colors[c] = color
            sum = sum + np.array(colors)
    return sum
    
def is_shadow(intersect, light, obj, normal):
    # get object position and scale
    pos = obj[0:3]
    scale = obj[3:6]
    
    light_pos = light[0:3]
    light_pos.append(1)
    
    # append 1 to pos since it is a point
    pos.append(1)
    
    # start from just off the object so that it does not instantly collide with itself (copy)
    intersection = (intersect[0:3] + np.array(normal.tolist()[0][0:3]) * 0.00001).tolist()
    intersection.append(0)
    intersection = np.array(intersection)
    ray = np.array(light_pos) - intersection
    
    # make a transformation matrix
    m_scale = np.matrix([[scale[0], 0, 0, 0], [0, scale[1], 0, 0], [0, 0, scale[2], 0], [0, 0, 0, 1]])
    m_pos = np.matrix([[1, 0, 0, pos[0]], [0, 1, 0, pos[1]], [0, 0, 1, pos[2]], [0, 0, 0, 1]])
    m = np.matmul(m_pos, m_scale)
    
    # inverse the transformation matrix
    inverse = np.linalg.inv(m)
    inverse_ray_s = np.matmul(inverse, intersection)
    inverse_ray_c = np.matmul(inverse, ray)
    
    
    # now we find whether we have an intersection and where
    # have to convert matrix to list and take element 0 to get an array
    a = get_magnitude(inverse_ray_c.tolist()[0]) ** 2
    b = float(np.dot(inverse_ray_s, inverse_ray_c.T))
    c = (get_magnitude(inverse_ray_s.tolist()[0]) ** 2) - 1.0
    
    # compute the inside of the square root to find how many intercepts (also can be re-used in t_prime computation)
    check = ((b**2) - (a * c))
    
    if check < 0.0:
        # no intercepts, apply local illumination
        return False
    elif check == 0:
        #one intercept
        t_prime = -(b / a)
    else:
        # two intercepts
        a1 = -(b / a) + np.sqrt(check) / a
        a2 = -(b / a) - np.sqrt(check) / a
        
        # get min if they> 0
        if a1 > 0 and a2 > 0:
            t_prime = min([a1, a2])
        elif a1 > 0:
            t_prime = a1
        elif a2 > 0:
            t_prime = a2
        else:
            return False
    # check if between intersection and light (0 and 1) for the vector
    if t_prime > 0.00001:
        return True
    return False
    
    
def find_intersections(obj, ray, ray_pos):
    # get object position and scale
    pos = obj[0:3]
    scale = obj[3:6]
    
    # append 1 to pos since it is a point
    pos.append(1)
    
    # make a transformation matrix
    m_scale = np.matrix([[scale[0], 0, 0, 0], [0, scale[1], 0, 0], [0, 0, scale[2], 0], [0, 0, 0, 1]])
    m_pos = np.matrix([[1, 0, 0, pos[0]], [0, 1, 0, pos[1]], [0, 0, 1, pos[2]], [0, 0, 0, 1]])
    m = np.matmul(m_pos, m_scale)
    
    # inverse the transformation matrix
    inverse = np.linalg.inv(m)
    inverse_ray_s = np.matmul(inverse, ray_pos)
    inverse_ray_c = np.matmul(inverse, ray)
    
    
    # now we find whether we have an intersection and where
    # have to convert matrix to list and take element 0 to get an array
    a = get_magnitude(inverse_ray_c.tolist()[0]) ** 2
    b = float(np.dot(inverse_ray_s, inverse_ray_c.T))
    c = (get_magnitude(inverse_ray_s.tolist()[0]) ** 2) - 1.0
    
    # compute the inside of the square root to find how many intercepts (also can be re-used in t_prime computation)
    check = ((b**2) - (a * c))
    
    if check < 0.0:
        # no intercepts
        return False
    elif check == 0:
        #one intercept
        t_prime = -(b / a)
    else:
        # two intercepts
        a1 = -(b / a) + np.sqrt(check) / a
        a2 = -(b / a) - np.sqrt(check) / a
        
        # get min if they> 0
        if a1 > 0 and a2 > 0:
            t_prime = min([a1, a2])
        elif a1 > 0:
            t_prime = a1
        elif a2 > 0:
            t_prime = a2
        else:
            return False
    # find the intersection using t_prime
    intersection = ray_pos + np.array(ray) * t_prime
    
    # find normal ( intersection - circle center )
    normal_m = np.matmul(inverse.T, (intersection - pos).T)
    
    # normalize normal
    normal = (normal_m.tolist()[0] / get_magnitude(normal_m.tolist()[0]))
    normal[3] = 0
    normal = np.matrix(normal)
    
    return [t_prime, normal]
    
def ray_trace(ray, ray_origin, depth):
    # copy ray origin so that the global is not affected
    ray_origin_copy = ray_origin + np.array(ray) * 0.000001
    if depth == 0:
        return np.array([0.0, 0.0, 0.0])
    
    t_primes, obj_intersections, normals = [], [], []
    for obj in s_array:
        intersection = find_intersections(obj, ray, ray_origin_copy)
        if isinstance(intersection, list):
            if intersection[0]:
                t_primes.append(intersection[0])
                normals.append(intersection[1])
                obj_intersections.append(obj)
    if not t_primes:
        if depth == 3:
            return np.array(back)
        else:
            return np.array([0.0, 0.0, 0.0])

    min_index = np.argmin(t_primes)
    t_prime = t_primes[min_index]
    normal = normals[min_index]
    
    intersected_obj = obj_intersections[min_index]
    intersect = t_prime * np.array(ray) + np.array(ray_origin_copy)
    reflected = (ray - 2 * np.dot(normal, np.array(ray).T) * normal).tolist()[0]
    reflected[3] = 0
    kr = intersected_obj[12]
    reflection_rgb = [0,0,0]
    if kr > 0:
        reflection_rgb = kr * ray_trace(reflected, intersect, depth - 1)
    colors = calculate_ambient(intersected_obj) + calculate_illumination(intersect, normal, t_prime, intersected_obj, ray) + reflection_rgb
    for x in range(0, len(colors)):
        if colors[x] > 1.0:
            colors[x] = 1
    return colors
    

image = {}
maxval = 255
# pixels from top to bottom and left to right
index = 0
for res_y in range(0,res[1]):
    for res_x in range(0,res[0]):
        ray = np.array([(left + (np.abs(right - left) * (res_x / res[0]))), (top - (np.abs(top - bottom) * (res_y / res[1]))), -near])
        
        # normalize the ray
        ray = ray / get_magnitude(ray)
        ray = [ray[0], ray[1], ray[2], 0]
        image[index] = ray_trace(ray, ray_origin, 3) * maxval
        index += 1
ray_trace([0,0,-1,0], ray_origin, 3) * maxval
# header
header = f'P3 {res[0]} {res[1]} {maxval}\n'
# write to ppm file
f=open(output,'wb')
f.write(bytearray(header, 'ascii'))
for x in range(0, len(image)):
    rgb = image[x]
    data = f'{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} '
    f.write(bytearray(data, 'ascii'))
f.close()