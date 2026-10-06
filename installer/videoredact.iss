; Inno Setup script for VideoRedact. Built by installer\build.ps1 which passes /DAppVersion=x.y.z
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define AppName "VideoRedact"
#define AppPublisher "VideoRedact project"
#define AppURL "https://github.com/wyattrossell/VideoRedact"

[Setup]
AppId={{7E1D4C1A-5B1E-4F2B-9D4E-VIDEOREDACT01}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
LicenseFile=..\LICENSE
OutputDir=Output
OutputBaseFilename=VideoRedact-{#AppVersion}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\VideoRedact.exe
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "assocproj"; Description: "Associate .vrproj project files with VideoRedact"; GroupDescription: "File associations:"

[Files]
Source: "..\dist\VideoRedact\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\VideoRedact.exe"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\VideoRedact.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.vrproj"; ValueType: string; ValueName: ""; ValueData: "VideoRedact.Project"; Flags: uninsdeletevalue; Tasks: assocproj
Root: HKA; Subkey: "Software\Classes\VideoRedact.Project"; ValueType: string; ValueName: ""; ValueData: "VideoRedact Project"; Flags: uninsdeletekey; Tasks: assocproj
Root: HKA; Subkey: "Software\Classes\VideoRedact.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\VideoRedact.exe,0"; Tasks: assocproj
Root: HKA; Subkey: "Software\Classes\VideoRedact.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\VideoRedact.exe"" ""%1"""; Tasks: assocproj

[Run]
Filename: "{app}\VideoRedact.exe"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
