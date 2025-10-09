import pkg_resources

installed = list(pkg_resources.working_set)
required = {r.key for d in installed for r in d.requires()}
top_level = sorted({d.project_name for d in installed if d.key not in required}, key=str.lower)

with open('requirements.txt', 'w', encoding='utf-8') as f:
    for name in top_level:
        f.write(f'{name}\n')