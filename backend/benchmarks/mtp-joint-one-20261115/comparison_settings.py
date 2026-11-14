"""Require the same validated execution environment for both candidates."""
import hashlib
import json


def load_settings(paths, checked):
    environments = []
    for path in paths:
        expected = checked.get(str(path.resolve()))
        data = path.read_bytes()
        if expected is None or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Environment evidence is missing or changed: ' + str(path))
        environment = json.loads(data)['environment']
        if not environment or any(not isinstance(k, str) or not isinstance(v, str)
                                  for k, v in environment.items()):
            raise ValueError('Invalid execution environment')
        environments.append(environment)
    if len(environments) != 2 or environments[0] != environments[1]:
        raise ValueError('Candidate execution environments differ')
    return environments[0].copy()
