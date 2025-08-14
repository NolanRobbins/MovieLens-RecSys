"""
Fix Data Mapping Consistency Issues
Regenerate user/movie mappings based on actual data
"""

import pandas as pd
import pickle
from pathlib import Path

def fix_data_mappings():
    """Fix user mapping consistency by regenerating mappings from actual data"""
    
    print("🔧 Fixing Data Mapping Consistency")
    print("=" * 40)
    
    # Load data
    data_path = Path('data/processed')
    train_df = pd.read_csv(data_path / 'train_data.csv')
    val_df = pd.read_csv(data_path / 'val_data.csv')
    
    print(f"📊 Loaded data:")
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Validation samples: {len(val_df):,}")
    
    # Combine all unique users and movies
    all_users = set(train_df['user_id']) | set(val_df['user_id'])
    all_movies = set(train_df['movie_id']) | set(val_df['movie_id'])
    
    print(f"\\n🔍 Found unique entities:")
    print(f"   Users: {len(all_users):,}")
    print(f"   Movies: {len(all_movies):,}")
    
    # Create new mappings
    user_to_index = {user_id: idx for idx, user_id in enumerate(sorted(all_users))}
    index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
    
    movie_to_index = {movie_id: idx for idx, movie_id in enumerate(sorted(all_movies))}
    index_to_movie = {idx: movie_id for movie_id, idx in movie_to_index.items()}
    
    # Create new mappings dictionary
    new_mappings = {
        'user_to_index': user_to_index,
        'index_to_user': index_to_user,
        'movie_to_index': movie_to_index,
        'index_to_movie': index_to_movie,
        'n_users': len(all_users),
        'n_movies': len(all_movies)
    }
    
    # Save the corrected mappings
    with open(data_path / 'data_mappings.pkl', 'wb') as f:
        pickle.dump(new_mappings, f)
    
    print(f"\\n✅ Fixed mappings saved:")
    print(f"   Users: {new_mappings['n_users']:,}")
    print(f"   Movies: {new_mappings['n_movies']:,}")
    
    # Verify consistency
    train_users = set(train_df['user_id'])
    val_users = set(val_df['user_id'])
    mapping_users = set(user_to_index.keys())
    
    train_movies = set(train_df['movie_id'])
    val_movies = set(val_df['movie_id'])
    mapping_movies = set(movie_to_index.keys())
    
    user_consistency = (train_users | val_users) == mapping_users
    movie_consistency = (train_movies | val_movies) == mapping_movies
    
    print(f"\\n🔍 Verification:")
    print(f"   User consistency: {'✅' if user_consistency else '❌'}")
    print(f"   Movie consistency: {'✅' if movie_consistency else '❌'}")
    
    if user_consistency and movie_consistency:
        print("\\n🎉 Data mappings fixed successfully!")
        return True
    else:
        print("\\n❌ Issues remain - manual investigation needed")
        return False

if __name__ == "__main__":
    success = fix_data_mappings()
    if success:
        print("\\n🚀 Ready to re-run validation suite")
    else:
        print("\\n⚠️ Manual data investigation required")