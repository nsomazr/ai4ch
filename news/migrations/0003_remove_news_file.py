# Generated by Django 5.0 on 2024-04-28 23:49

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("news", "0002_alter_news_thumbnail"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="news",
            name="file",
        ),
    ]