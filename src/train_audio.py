import os
import tensorflow as tf
from datetime import datetime
from swin_transformer import build_swin_tiny


# =========================
# CONFIG
# =========================
class Config:
	DATA_ROOT = '/kaggle/input/datasets/bishertello/asvspoof-21-df-cqt/my_dataset'

	TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
	VAL_DIR = os.path.join(DATA_ROOT, 'validation')
	TEST_DIR = os.path.join(DATA_ROOT, 'test')

	CHECKPOINT_DIR = '/kaggle/working/models/audio_checkpoints'
	IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

	LABELS = {'real': 0, 'fake': 1}

	epochs = 3
	batch_size = 16
	lr = 1e-4


# =========================
# DATA LOADER
# =========================
def _collect_image_paths(split_root, class_map):
	paths, labels = [], []

	if not os.path.exists(split_root):
		return paths, labels

	for class_name, label in class_map.items():
		class_dir = os.path.join(split_root, class_name)
		if not os.path.exists(class_dir):
			continue

		for name in os.listdir(class_dir):
			if name.lower().endswith(Config.IMAGE_EXTS):
				paths.append(os.path.join(class_dir, name))
				labels.append(label)

	return paths, labels


def load_and_prepare_data():
	print('📂 Loading audio CQT dataset paths...')

	train_paths, train_labels = _collect_image_paths(Config.TRAIN_DIR, Config.LABELS)
	val_paths, val_labels = _collect_image_paths(Config.VAL_DIR, Config.LABELS)
	test_paths, test_labels = _collect_image_paths(Config.TEST_DIR, Config.LABELS)

	output = {
		'train_paths': train_paths,
		'train_labels': train_labels,
		'val_paths': val_paths,
		'val_labels': val_labels,
		'test_paths': test_paths,
		'test_labels': test_labels
	}

	print(f"Train samples: {len(output['train_paths'])}")
	print(f"Val samples: {len(output['val_paths'])}")
	print(f"Test samples: {len(output['test_paths'])}")

	return output


# =========================
# TRAINING FUNCTION
# =========================
def run_training_session(
		data,
		epochs=Config.epochs,
		batch_size=Config.batch_size,
		lr=Config.lr):

	os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

	def _decode(path, label):
		img = tf.image.decode_image(
			tf.io.read_file(path),
			channels=3,
			expand_animations=False
		)
		img = tf.image.resize(img, (224, 224))
		img = tf.cast(img, tf.float32) / 255.0
		return img, label

	train_ds = (
		tf.data.Dataset
		.from_tensor_slices((data['train_paths'], data['train_labels']))
		.shuffle(10000)
		.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
		.batch(batch_size)
		.prefetch(tf.data.AUTOTUNE)
	)

	val_ds = (
		tf.data.Dataset
		.from_tensor_slices((data['val_paths'], data['val_labels']))
		.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
		.batch(batch_size)
		.prefetch(tf.data.AUTOTUNE)
	)

	test_ds = None
	if data.get('test_paths'):
		test_ds = (
			tf.data.Dataset
			.from_tensor_slices((data['test_paths'], data['test_labels']))
			.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
			.batch(batch_size)
			.prefetch(tf.data.AUTOTUNE)
		)

	model = build_swin_tiny(
		input_shape=(224, 224, 3),
		num_classes=2
	)

	model.compile(
		optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
	)

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			os.path.join(
				Config.CHECKPOINT_DIR,
				f'audio_best_model_{timestamp}.h5'
			),
			monitor='val_auc',
			save_best_only=True,
			mode='max'
		)
	]

	print(f'🚀 Training audio model for {epochs} epochs...')

	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs,
		callbacks=callbacks
	)

	test_metrics = None
	if test_ds is not None:
		print('🧪 Evaluating on test split...')
		test_metrics = model.evaluate(test_ds, return_dict=True)
		print(f'Test metrics: {test_metrics}')

	return model, history, test_metrics
