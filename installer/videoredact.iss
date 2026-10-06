; Inno Setup script for VideoRedact. Built by installer\build.ps1 which passes /DAppVersion=x.y.z
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "VideoRedact"
#define AppPublisher "VideoRedact project"
#define AppURL "https://github.com/wyattrossell/VideoRedact"

[Setup]
; AppId must never change: it is how upgrades (including the in-app auto-update) find the existing install.
AppId={{B5E7C6D2-3A4F-4E0B-9C21-5D8F1A2B3C4D}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
VersionInfoVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
LicenseFile=..\LICENSE
OutputDir=..\Published
OutputBaseFilename=VideoRedact-{#AppVersion}-Setup
SetupIconFile=videoredact.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; "lowest" lets a standard user install/update into %LocalAppData%\Programs without UAC,
; which is what the in-app auto-update needs. Admins can still pick "all users" from the dialog.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
UninstallDisplayIcon={app}\VideoRedact.exe
ChangesAssociations=yes
CloseApplications=yes
RestartApplications=yes
DisableProgramGroupPage=yes

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

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
