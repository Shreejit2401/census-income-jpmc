import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class SegmentationModel:
    def __init__(self, processed_dir, models_dir, raw_data_path=None):
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_path = raw_data_path
        
        self.RANDOM_STATE = 2401
        self.pca = None
        self.kmeans = None
        self.X_pca = None
        self.clusters = None
        
        self._load_data()
    
    def _load_data(self):
        self.X_seg = np.load(self.processed_dir / "Segmentation.npy")
        self.X_seg_df = pd.read_csv(self.processed_dir / "Segmentation_data.csv")
        
        seg_target_path = self.processed_dir / "Segmented_Data_Target.csv"
        if seg_target_path.exists():
            temp_df = pd.read_csv(seg_target_path)
            if 'label' in temp_df.columns:
                self.X_seg_df['label'] = temp_df['label']
    
    def fit_pca(self, target_variance=0.80, n_components=None):
        pca_full = PCA(random_state=self.RANDOM_STATE)
        pca_full.fit(self.X_seg)
        
        explained = pca_full.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        
        if n_components is None:
            n_components = np.argmax(cumulative >= target_variance) + 1
        
        self.pca = PCA(n_components=n_components, random_state=self.RANDOM_STATE)
        self.X_pca = self.pca.fit_transform(self.X_seg)
        
        print(f"PCA: {n_components} components explain {cumulative[n_components-1]:.2%} variance")
        return self.X_pca
    
    def determine_optimal_clusters(self, k_range=range(2, 12)):
        inertias = []
        silhouettes = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.RANDOM_STATE, n_init=10)
            km.fit(self.X_pca)
            inertias.append(km.inertia_)
            
            sil = silhouette_score(self.X_pca, km.labels_, sample_size=10000, random_state=self.RANDOM_STATE)
            silhouettes.append(sil)
            print(f"k={k}  inertia={km.inertia_:.0f}  silhouette={sil:.4f}")
        
        return inertias, silhouettes
    
    def fit_kmeans(self, n_clusters=6):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.RANDOM_STATE, n_init=10)
        self.clusters = self.kmeans.fit_predict(self.X_pca)
        
        self.X_seg_df['cluster'] = self.clusters
        
        cluster_dist = pd.Series(self.clusters).value_counts().sort_index()
        print(f"Cluster distribution:\n{cluster_dist}")
        
        return self.clusters
    
    def create_cluster_profiles(self, raw_data_path=None):
        if raw_data_path:
            data = pd.read_csv(raw_data_path)
        else:
            data = pd.read_csv(self.processed_dir / "Raw_Data.csv")
        
        seg_cols = [
            'age', 'education', 'weeks worked in year',
            'capital gains', 'dividends from stocks',
            'family members under 18', 'own business or self employed',
            'marital stat', 'class of worker',
            'full or part time employment stat', 'tax filer stat',
            'detailed household summary in household', 'major occupation code',
        ]
        
        data_seg_raw = data[seg_cols].copy()
        data_seg_raw['cluster'] = self.clusters
        if 'label' in data.columns:
            data_seg_raw['label'] = data['label'].values
        
        num_profile = data_seg_raw.groupby('cluster').agg({
            'age': 'mean',
            'weeks worked in year': 'mean',
            'capital gains': 'mean',
            'dividends from stocks': 'mean',
        }).round(2)
        
        cat_cols_raw = [
            'education', 'marital stat', 'class of worker',
            'full or part time employment stat', 'tax filer stat',
            'detailed household summary in household', 'major occupation code',
            'family members under 18'
        ]
        
        cat_profile = data_seg_raw.groupby('cluster')[cat_cols_raw].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        
        return data_seg_raw, num_profile, cat_profile
    
    def assign_segment_names(self, cluster_names=None):
        if cluster_names is None:
            cluster_names = {
                0: 'Retired / Elderly',
                1: 'Children / Dependents',
                2: 'Working Middle Class',
                3: 'High Earners',
                4: 'Wealthy Investors',
                5: 'Young Working Adults',
            }
        
        self.cluster_names = cluster_names
        self.X_seg_df['segment'] = self.X_seg_df['cluster'].map(cluster_names)
        
        return self.X_seg_df
    
    def generate_segment_summary(self):
        summary = self.X_seg_df.groupby('segment').agg(
            size=('cluster', 'count'),
            pct_of_pop=('cluster', lambda x: round(len(x) / len(self.X_seg_df) * 100, 2))
        ).reset_index()
        
        if 'label' in self.X_seg_df.columns:
            summary['pct_over_50k'] = self.X_seg_df.groupby('segment')['label'].apply(
                lambda x: round((x != 0).mean() * 100, 2)
            ).values
        
        summary = summary.sort_values('size', ascending=False)
        
        print("\nSegment Summary:")
        print(summary)
        
        return summary
    
    def save_models(self):
        joblib.dump(self.kmeans, self.models_dir / 'kmeans_segmentation_model.joblib', compress=3)
        joblib.dump(self.pca, self.models_dir / 'segmentation_pca_model.joblib', compress=3)
        
        np.save(self.processed_dir / 'segmentation_X_pca.npy', self.X_pca)
        
        print("\nSaved models:")
        print(f"  {self.models_dir / 'kmeans_segmentation_model.joblib'}")
        print(f"  {self.models_dir / 'segmentation_pca_model.joblib'}")
    
    def save_results(self, segment_summary):
        segment_summary.to_csv(self.processed_dir / 'segmentation_summary.csv', index=False)
        
        seg_assignments = self.X_seg_df.copy()
        seg_assignments.to_pickle(self.processed_dir / 'segmentation_assignments.pkl')
        
        print("\nSaved results:")
        print(f"  {self.processed_dir / 'segmentation_X_pca.npy'}")
        print(f"  {self.processed_dir / 'segmentation_summary.csv'}")
        print(f"  {self.processed_dir / 'segmentation_assignments.pkl'}")


def train_segmentation_pipeline(processed_dir, models_dir, raw_data_path=None):
    model = SegmentationModel(processed_dir, models_dir, raw_data_path)
    
    print("="*80)
    print("SEGMENTATION PIPELINE")
    print("="*80)
    
    print("\nStep 1: Fitting PCA...")
    X_pca = model.fit_pca(target_variance=0.80, n_components=7)
    
    print("\nStep 2: Determining optimal number of clusters...")
    inertias, silhouettes = model.determine_optimal_clusters(k_range=range(2, 12))
    
    print("\nStep 3: Fitting KMeans with k=6 (optimal)...")
    clusters = model.fit_kmeans(n_clusters=6)
    
    print("\nStep 4: Assigning segment names...")
    model.assign_segment_names()
    
    print("\nStep 5: Creating cluster profiles...")
    data_seg_raw, num_profile, cat_profile = model.create_cluster_profiles(raw_data_path)
    
    print("\nNumeric Profile:")
    print(num_profile)
    print("\nCategorical Profile:")
    print(cat_profile)
    
    print("\nStep 6: Generating segment summary...")
    segment_summary = model.generate_segment_summary()
    
    print("\nStep 7: Saving models and results...")
    model.save_models()
    model.save_results(segment_summary)
    
    print("\n" + "="*80)
    print("SEGMENTATION COMPLETE")
    print("="*80)
    
    print("\nCluster Mapping:")
    for cluster_id, name in model.cluster_names.items():
        print(f"  {cluster_id}: {name}")
    
    return model, segment_summary


if __name__ == "__main__":
    processed_dir = r"D:\Projects\TakeHomeProject\data\processed"
    models_dir = r"D:\Projects\TakeHomeProject\outputs\models"
    raw_data_path = r"D:\Projects\TakeHomeProject\data\processed\Raw_Data.csv"
    
    model, summary = train_segmentation_pipeline(processed_dir, models_dir, raw_data_path)
    print("\nTraining completed.")
