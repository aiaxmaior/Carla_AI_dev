#!/usr/bin/env python3
"""
Q-DRIVE Cortex Modular Structure Migration Script
================================================

This script safely migrates your existing Q-DRIVE Cortex project to a modular structure.
It creates backups, moves files, updates imports, and verifies everything works.

Usage: python migrate_to_modular.py [--dry-run] [--backup-dir BACKUP_DIR]
"""

import os
import shutil
import re
import glob
import argparse
from pathlib import Path
import subprocess
import sys

class ModularMigration:
    
    def __init__(self, backup_dir="backup_before_modular"):
        self.backup_dir = backup_dir
        self.project_root = Path.cwd()
        
        # Define the migration mapping
        self.file_migrations = {
            # Files to move to Utility/Monitor/
            "DynamicMonitor.py": "Utility/Monitor/DynamicMonitor.py",
            "monitor_config.py": "Utility/Monitor/monitor_config.py",
            
            # Files to move to Utility/UI/ (if they exist)
            "dynamic_mapping.py": "Utility/UI/dynamic_mapping.py",
            
            # Files to move to Core/Sensors/ (if they exist)
            "Sensors.py": "Core/Sensors/Sensors.py",
            
            # Files to move to Core/Vision/ (if they exist) 
            "VisionPerception.py": "Core/Vision/VisionPerception.py",
            
            # Files to move to Core/Simulation/ (if they exist)
            "MVD.py": "Core/Simulation/MVD.py",
        }
        
        # Import statement replacements
        self.import_replacements = {
            r'from Utility.Monitor import Utility.Monitor.DynamicMonitor as DynamicMonitor': 'from Utility.Monitor import Utility.Monitor.DynamicMonitor as DynamicMonitor',
            r'import Utility.Monitor.DynamicMonitor as DynamicMonitor': 'import Utility.Monitor.DynamicMonitor as DynamicMonitor',
            r'from Core.Sensors.Sensors import': 'from Core.Sensors.Sensors import',
            r'from Core.Vision.VisionPerception import': 'from Core.Vision.VisionPerception import', 
            r'from Core.Simulation.MVD import': 'from Core.Simulation.MVD import',
            r'from Utility.UI import dynamic_mapping': 'from Utility.UI from Utility.UI import dynamic_mapping',
        }
    
    def create_backup(self):
        """Create a full backup of the current project state."""
        print(f"🔄 Creating backup in {self.backup_dir}/...")
        
        if os.path.exists(self.backup_dir):
            print(f"⚠️  Backup directory {self.backup_dir} already exists!")
            response = input("Overwrite? (y/n): ").strip().lower()
            if response != 'y':
                print("❌ Migration cancelled.")
                return False
            shutil.rmtree(self.backup_dir)
        
        shutil.copytree(self.project_root, self.backup_dir, 
                       ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        print("✅ Backup created successfully!")
        return True
    
    def create_directory_structure(self):
        """Create the new modular directory structure."""
        print("📁 Creating modular directory structure...")
        
        directories = [
            "Utility",
            "Utility/Monitor", 
            "Utility/Hardware",
            "Utility/Audio",
            "Utility/UI",
            "Utility/Data",
            "Core",
            "Core/Sensors",
            "Core/Vision", 
            "Core/Controls",
            "Core/Simulation"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
            # Create __init__.py file
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                with open(init_file, 'w') as f:
                    f.write(f'"""{directory.replace("/", ".")} package for Q-DRIVE Cortex"""\n')
        
        print("✅ Directory structure created!")
    
    def migrate_files(self, dry_run=False):
        """Move files to their new locations."""
        print("📦 Migrating files to new structure...")
        
        moved_files = []
        
        for old_path, new_path in self.file_migrations.items():
            if os.path.exists(old_path):
                if dry_run:
                    print(f"  [DRY-RUN] Would move: {old_path} → {new_path}")
                else:
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    
                    # Move the file
                    shutil.move(old_path, new_path)
                    moved_files.append((old_path, new_path))
                    print(f"  ✅ Moved: {old_path} → {new_path}")
            else:
                print(f"  ⚠️  File not found: {old_path}")
        
        return moved_files
    
    def update_imports(self, dry_run=False):
        """Update import statements in Python files."""
        print("🔧 Updating import statements...")
        
        # Define directories to EXCLUDE from import updates
        excluded_dirs = {
            'Backups', 'backups', 'backup', 'backup_before_modular',
            'Main_archive_6.24', 'Modular_Scripts_1', 'test_folder', 
            '1851', 'a', 'archive', 'archives', 'old', 'deprecated',
            '__pycache__', '.git', '.vscode', 'venv', 'env'
        }
        
        # Define files that SHOULD be updated (whitelist approach)
        files_to_update = [
            # Core project files in root
            'Main.py', 'HUD.py', 'World.py', 'controls_queue.py',
            'Helpers.py', 'EventManager.py', 'Sensors.py', 'VisionPerception.py',
            'MVD.py', 'dynamic_mapping.py', 'sys_task.py',
            
            # New utility files (after they're moved)
            'Utility/Monitor/DynamicMonitor.py',
            'Utility/Monitor/monitor_config.py',
            'Utility/UI/dynamic_mapping.py',
            'Core/Sensors/Sensors.py',
            'Core/Vision/VisionPerception.py',
            'Core/Simulation/MVD.py',
        ]
        
        # Find Python files, but filter them intelligently
        all_python_files = glob.glob("*.py") + glob.glob("**/*.py", recursive=True)
        
        # Filter out excluded directories and backup files
        python_files = []
        for file_path in all_python_files:
            file_path = file_path.replace('\\', '/')  # Normalize path separators
            
            # Skip if in excluded directory
            path_parts = file_path.split('/')
            if any(part in excluded_dirs for part in path_parts):
                continue
                
            # Skip backup files by name pattern
            filename = os.path.basename(file_path)
            if any(pattern in filename.lower() for pattern in ['backup', 'archive', '_old', '_original', 'deprecated']):
                continue
                
            # For root files, only include if in whitelist OR if it doesn't contain excluded patterns
            if '/' not in file_path:  # Root level file
                if file_path in files_to_update or not any(excluded in file_path.lower() for excluded in ['test', 'backup', 'archive', 'old']):
                    python_files.append(file_path)
            else:
                # For subdirectory files, be more selective
                if file_path in files_to_update:
                    python_files.append(file_path)
        
        print(f"📋 Found {len(python_files)} files to check for import updates")
        if dry_run and len(python_files) > 10:
            print("   First 10 files:")
            for f in python_files[:10]:
                print(f"     • {f}")
            print(f"   ... and {len(python_files) - 10} more")
        
        updated_files = []
        
        for file_path in python_files:
            if self._update_file_imports(file_path, dry_run):
                updated_files.append(file_path)
        
        if updated_files:
            print(f"✅ Updated imports in {len(updated_files)} files")
        else:
            print("ℹ️  No import updates needed")
            
        return updated_files
    
    def _update_file_imports(self, file_path, dry_run=False):
        """Update imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            changes_made = False
            
            # Apply each replacement
            for old_pattern, new_import in self.import_replacements.items():
                if re.search(old_pattern, content):
                    content = re.sub(old_pattern, new_import, content)
                    changes_made = True
            
            if changes_made:
                if dry_run:
                    print(f"  [DRY-RUN] Would update imports in: {file_path}")
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Updated imports in: {file_path}")
                
                return True
            
        except Exception as e:
            print(f"  ❌ Error updating {file_path}: {e}")
        
        return False
    
    def create_init_files(self):
        """Create proper __init__.py files with imports."""
        print("📄 Creating enhanced __init__.py files...")
        
        # Utility/__init__.py
        utility_init = """'''
Q-DRIVE Cortex Utility Package
=============================

This package contains utility modules for the Q-DRIVE Cortex driving simulator.
'''

__version__ = "1.0.0"

# Import commonly used utilities
try:
    from .Monitor.DynamicMonitor import Utility.Monitor.DynamicMonitor as DynamicMonitor
    from .Monitor.monitor_config import MonitorConfigUtility
except ImportError:
    pass

__all__ = ['DynamicMonitor', 'MonitorConfigUtility']
"""
        
        # Utility/Monitor/__init__.py  
        monitor_init = """'''Monitor Management Utilities'''

from .DynamicMonitor import Utility.Monitor.DynamicMonitor as DynamicMonitor
from .monitor_config import MonitorConfigUtility

__all__ = ['DynamicMonitor', 'MonitorConfigUtility']
"""
        
        init_files = {
            "Utility/__init__.py": utility_init,
            "Utility/Monitor/__init__.py": monitor_init,
        }
        
        for file_path, content in init_files.items():
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"  ✅ Created: {file_path}")
    
    def test_imports(self):
        """Test that the new import structure works."""
        print("🧪 Testing new import structure...")
        
        test_imports = [
            "from Utility.Monitor import Utility.Monitor.DynamicMonitor as DynamicMonitor",
            "from Utility.Monitor.DynamicMonitor import Utility.Monitor.DynamicMonitor as DynamicMonitor",
        ]
        
        success_count = 0
        for import_stmt in test_imports:
            try:
                exec(import_stmt)
                print(f"  ✅ {import_stmt}")
                success_count += 1
            except ImportError as e:
                print(f"  ❌ {import_stmt} - {e}")
        
        return success_count == len(test_imports)
    
    def run_migration(self, dry_run=False):
        """Run the complete migration process."""
        print("🚀 Starting Q-DRIVE Cortex modular migration...")
        
        if not dry_run:
            if not self.create_backup():
                return False
        
        self.create_directory_structure()
        
        moved_files = self.migrate_files(dry_run)
        updated_files = self.update_imports(dry_run)
        
        if not dry_run:
            self.create_init_files()
            
            # Test the migration
            if self.test_imports():
                print("\n🎉 Migration completed successfully!")
                print(f"📁 Backup saved in: {self.backup_dir}/")
                print("🎯 Next steps:")
                print("  1. Test your application: python Main.py")
                print("  2. Run monitor config: python Utility/Monitor/monitor_config.py --test")
                print("  3. If issues occur, restore from backup")
                return True
            else:
                print("\n⚠️  Migration completed but imports are failing.")
                print("Check the error messages above and fix any issues.")
                return False
        else:
            print(f"\n📋 DRY-RUN SUMMARY:")
            print(f"  Files to move: {len(moved_files)}")
            print(f"  Files to update: {len(updated_files)}")
            print("  Run without --dry-run to execute the migration.")
            return True

def main():
    parser = argparse.ArgumentParser(description="Migrate Q-DRIVE Cortex to modular structure")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    parser.add_argument("--backup-dir", default="backup_before_modular",
                       help="Directory name for backup (default: backup_before_modular)")
    
    args = parser.parse_args()
    
    if not os.path.exists("Main.py"):
        print("❌ This doesn't appear to be a Q-DRIVE Cortex project directory.")
        print("Please run this script from your project root (where Main.py is located).")
        return 1
    
    migration = ModularMigration(args.backup_dir)
    success = migration.run_migration(args.dry_run)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())