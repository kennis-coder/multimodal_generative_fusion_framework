# -*- encoding: utf-8 -*-

# @Date    : 6/28/19
# @Author  : Kennis Yu

from collections import deque


def lookup(source, target_name, max_depth):
    """
    lookup a target in a python object like module, packages, class, etc
    :param source: module, package and class
    :param target_name: property, method
    :param max_depth
    :return:
    """
    if not hasattr(source, '__name__'):
        return

    assert source.__name__ != target_name, '{} cannot be the source name!'.format(target_name)

    ancestor = (source, [source.__name__])
    queue = deque()
    queue.append(ancestor)

    accessed = set()

    flag = 1

    while queue and flag:
        c_ancestor = queue.pop()
        c_obj = c_ancestor[0]
        accessed.add(c_obj)

        if len(c_ancestor[1]) > max_depth:
            continue

        for n_name in dir(c_obj):
            n_obj = getattr(c_obj, n_name)

            if n_name == target_name:
                print('.'.join(c_ancestor[1]) + '.' + n_name)
                flag = 0
                break

            if type(n_obj).__name__ != 'module':
                continue

            print(n_name)

            temp = []
            temp.extend(c_ancestor[1])
            temp.append(n_name)
            n_ancestor = (n_obj, temp)
            queue.append(n_ancestor)


