@echo off
echo Creating knowledge directory...
mkdir "C:\Dev\coding_agent\qwen30b_chat\dist\Qwen30B_Chat\knowledge" 2>nul
echo Copying knowledge file...
copy "C:\Dev\coding_agent\qwen30b_chat\knowledge\company_rules.txt" "C:\Dev\coding_agent\qwen30b_chat\dist\Qwen30B_Chat\knowledge\" /Y
echo Done!
pause
