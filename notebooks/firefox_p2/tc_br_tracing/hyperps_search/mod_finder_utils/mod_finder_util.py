
def add_modules_origin_search_path():
    import sys
    
    MODULES_ORIGIN_SEARCH_PATH = '../../../..'
    
    if MODULES_ORIGIN_SEARCH_PATH not in sys.path:
        sys.path.append(MODULES_ORIGIN_SEARCH_PATH)