# Generated by Django 5.0 on 2024-01-12 13:58

import ckeditor_uploader.fields
import django.db.models.deletion
import django.utils.timezone
import django_resized.forms
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="News",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("title", models.CharField(max_length=500)),
                (
                    "header_image",
                    django_resized.forms.ResizedImageField(
                        blank=True,
                        crop=["middle", "center"],
                        force_format="PNG",
                        keep_meta=True,
                        quality=100,
                        scale=0.5,
                        size=[1400, 600],
                        upload_to="news",
                    ),
                ),
                (
                    "thumbnail",
                    django_resized.forms.ResizedImageField(
                        blank=True,
                        crop=["middle", "center"],
                        force_format="PNG",
                        keep_meta=True,
                        quality=100,
                        scale=0.5,
                        size=[1080, 1080],
                        upload_to="news",
                    ),
                ),
                ("description", models.TextField(max_length=200)),
                (
                    "body",
                    ckeditor_uploader.fields.RichTextUploadingField(
                        blank=True, null=True
                    ),
                ),
                (
                    "file",
                    models.FileField(blank=True, max_length=500, upload_to="news"),
                ),
                ("status", models.IntegerField(default=0)),
                ("publish", models.IntegerField(default=0)),
                ("reject", models.IntegerField(default=0)),
                ("thematic_area", models.IntegerField(default=0)),
                ("slug", models.SlugField(max_length=255, unique=True)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("updated_at", models.DateTimeField(default=django.utils.timezone.now)),
                (
                    "publisher",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]