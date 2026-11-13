"""Identity classification only: no GPU query or benchmark launch."""
import unittest
from environment import desktop_identity_allowed

PACKAGE='C:/Program Files/WindowsApps/OpenAI.Codex_26.1103.4239.0_x64__2p2nqsd0c76g0/app/'
HELPER='resources/cua_node/bin/node_modules/@oai/sky/bin/windows/swift/x64/codex-computer-use-swift.exe'
SYSTEM='C:/Windows/SystemApps/MicrosoftWindows.Client.CBS_cw5n1h2txyewy/CrossDeviceResume.exe'

class Identity(unittest.TestCase):
    def check(self,path,expected,kind='C+G'):
        self.assertEqual(desktop_identity_allowed(dict(name=path,type=kind)),expected)
    def test_installed_surfaces(self):
        for path in (PACKAGE+'ChatGPT.exe',PACKAGE+HELPER,SYSTEM):
            self.check(path,True);self.check(path.replace('/','\\').upper(),True)
    def test_compute_not_exempt(self):
        for path in (PACKAGE+'ChatGPT.exe',PACKAGE+HELPER,SYSTEM,'C:/Windows/explorer.exe'):
            self.check(path,False,'C')
    def test_lookalikes(self):
        for path in ('C:/Temp/ChatGPT.exe','C:/Temp/codex-computer-use-swift.exe',
                     'C:/Temp/CrossDeviceResume.exe',PACKAGE.replace('OpenAI.Codex_','Fake.Codex_')+'ChatGPT.exe',
                     PACKAGE+'other/ChatGPT.exe',SYSTEM+'.exe',PACKAGE+'ChatGPT.exe/child'):
            self.check(path,False)
    def test_noncritical_apps_not_exempt(self):
        for path in ('C:/Program Files/Mozilla Firefox/firefox.exe','D:/thorium/BIN/thorium.exe',
                     'C:/Program Files/Cloudflare/Cloudflare WARP/Cloudflare WARP.exe','C:/tools/fdm.exe'):
            self.check(path,False)
    def test_existing_development_surface(self):
        self.check('C:/tools/Code.exe',True)

if __name__=='__main__':unittest.main()
